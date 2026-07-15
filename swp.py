import os
import gc
import random
import logging
import argparse
import copy
from typing import Any, Dict, List

import torch
import pynvml
import numpy as np
import pandas as pd

from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from config.config_regression import ConfigRegression

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Avoid tokenizer / dataloader deadlock.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ============================================================
# Fixed setting for this sensitivity experiment
# ============================================================
DATASET_NAME = "simsv2"
TRAIN_MODE = "regression"
DEFAULT_SEED = 1111

# Main-table SIMS-v2 hyperparameters.
# For every run, restore these values, then change only one hyperparameter.
BASE_SIMSV2_HPARAMS = {
    "cib_scale": 0.15,
    "beta": 1.2,
    "modality_dropout": 0.3,
    "pseudo_tokens": 8,
}

# One-factor-at-a-time sweep grid.
SWEEP_GRID = {
    "cib_scale": [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
    "beta": [0.10, 0.30, 0.50, 0.70, 1.00, 1.20, 1.50],
    "modality_dropout": [0.00, 0.10, 0.20, 0.30, 0.40, 0.50],
    "pseudo_tokens": [4, 8, 16, 32],
}

INT_HPARAMS = {"pseudo_tokens", "subspace_dim", "qformer_layers", "cnn_kernel", "cnn_stride"}


# ============================================================
# Reproducibility
# ============================================================
def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Logging
# ============================================================
def set_log(args):
    os.makedirs(args.log_dir, exist_ok=True)
    log_file_path = os.path.join(
        args.log_dir,
        f"hparam_sensitivity_{args.modelName}_{DATASET_NAME}_seed{args.seed}_full.log",
    )

    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter_file = logging.Formatter(
        "%(asctime)s:%(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)

    formatter_stream = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file_path}")
    return logger


# ============================================================
# Utilities
# ============================================================
def sanitize_value(v: Any) -> str:
    """Make a value safe for run_tag / filename."""
    if isinstance(v, float):
        s = f"{v:g}"
    else:
        s = str(v)
    s = s.replace(".", "p").replace("-", "m")
    s = s.replace("/", "_").replace(" ", "")
    return s


def cast_sweep_value(param: str, value: Any) -> Any:
    if param in INT_HPARAMS:
        return int(float(value))
    return float(value)


def get_device(args):
    if len(args.gpu_ids) == 0 and torch.cuda.is_available():
        try:
            pynvml.nvmlInit()
            dst_gpu_id, min_mem_used = 0, 1e16
            # Default auto-select only checks GPU 6, following the previous script.
            for g_id in [6]:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    if meminfo.used < min_mem_used:
                        min_mem_used = meminfo.used
                        dst_gpu_id = g_id
                except Exception:
                    continue
            args.gpu_ids.append(dst_gpu_id)
            logger.info(f"Find gpu: {dst_gpu_id}, used memory: {min_mem_used}!")
        except Exception as e:
            logger.info(f"GPU auto-select failed: {e}. Defaulting to GPU 0.")
            args.gpu_ids.append(0)

    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    return torch.device(f"cuda:{int(args.gpu_ids[0])}" if using_cuda else "cpu")


def apply_base_hparams(args):
    """Restore the main-table SIMS-v2 hyperparameters before each single-factor run."""
    for key, value in BASE_SIMSV2_HPARAMS.items():
        setattr(args, key, value)
    return args


def apply_single_sweep(args, sweep_param: str, sweep_value: Any):
    """One-factor-at-a-time: restore base values, then change only current hyperparameter."""
    apply_base_hparams(args)
    sweep_value = cast_sweep_value(sweep_param, sweep_value)
    setattr(args, sweep_param, sweep_value)
    return args


def build_sweep_plan(args) -> List[Dict[str, Any]]:
    """
    Build one-factor-at-a-time sensitivity plan.
    If args.sweep_param == 'all', run all four curves.
    Otherwise run a single curve, with optional custom values.
    """
    if args.sweep_param != "all":
        if args.sweep_values is not None:
            values = []
            for item in args.sweep_values.split(","):
                item = item.strip()
                if item == "":
                    continue
                values.append(cast_sweep_value(args.sweep_param, item))
        else:
            values = SWEEP_GRID.get(args.sweep_param)
            if values is None:
                raise ValueError(f"Unknown sweep_param: {args.sweep_param}")
        return [
            {
                "sweep_param": args.sweep_param,
                "sweep_value": cast_sweep_value(args.sweep_param, v),
            }
            for v in values
        ]

    plan = []
    for param, values in SWEEP_GRID.items():
        for value in values:
            plan.append(
                {
                    "sweep_param": param,
                    "sweep_value": cast_sweep_value(param, value),
                }
            )
    return plan


def load_finished_run_tags(summary_path: str) -> set:
    """Used for resume. Failed rows are not considered finished."""
    if not os.path.exists(summary_path):
        return set()
    try:
        old_df = pd.read_csv(summary_path)
        if "run_tag" not in old_df.columns:
            return set()
        if "error" in old_df.columns:
            ok_df = old_df[old_df["error"].isna() | (old_df["error"].astype(str) == "")]
        else:
            ok_df = old_df
        return set(ok_df["run_tag"].astype(str).tolist())
    except Exception:
        return set()


def load_existing_summary_rows(summary_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(summary_path):
        return []
    try:
        return pd.read_csv(summary_path).to_dict("records")
    except Exception:
        return []


def copy_runtime_args(src, dst):
    """Keep command-line runtime/path settings after ConfigRegression reconstructs args."""
    dst.root_dataset_dir = src.root_dataset_dir
    dst.model_save_dir = src.model_save_dir
    dst.res_save_dir = src.res_save_dir
    dst.pretrain_LM = src.pretrain_LM
    dst.num_workers = src.num_workers
    dst.gpu_ids = copy.deepcopy(src.gpu_ids)
    dst.tune_mode = src.tune_mode
    dst.is_tune = src.is_tune
    dst.log_dir = src.log_dir
    return dst


# ============================================================
# One run
# ============================================================
def run_one_setting(args):
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.res_save_dir, exist_ok=True)

    run_tag = getattr(args, "run_tag", "default")
    args.model_save_path = os.path.join(
        args.model_save_dir,
        f"{args.modelName}-{args.datasetName}-{args.train_mode}-{run_tag}.pth",
    )

    device = get_device(args)
    args.device = device

    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)

    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_params} || "
        f"trainable%: {100 * trainable_params / all_params:.6f}"
    )

    atio = ATIO().getTrain(args)
    atio.do_train(model, dataloader)

    if os.path.exists(args.model_save_path):
        checkpoint = torch.load(args.model_save_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint, strict=False)
    else:
        logger.warning(f"Best checkpoint not found at {args.model_save_path}. Testing current model.")

    model.to(device)
    if args.tune_mode:
        results = atio.do_test(model, dataloader["valid"], mode="VALID")
    else:
        results = atio.do_test(model, dataloader["test"], mode="TEST")

    del model
    del atio
    del dataloader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return results


# ============================================================
# Main sensitivity loop
# ============================================================
def run_sensitivity(init_args):
    init_args.datasetName = DATASET_NAME
    init_args.train_mode = TRAIN_MODE
    seed = int(init_args.seed)
    init_args.seed = seed
    init_args.seeds = [seed]

    set_log(init_args)
    setup_seed(seed)

    os.makedirs(init_args.res_save_dir, exist_ok=True)
    os.makedirs(init_args.model_save_dir, exist_ok=True)

    summary_path = os.path.join(
        init_args.res_save_dir,
        f"hparam_sensitivity_simsv2_seed{seed}_full.csv",
    )

    sweep_plan = build_sweep_plan(init_args)
    logger.info(f"Total sensitivity runs in plan: {len(sweep_plan)}")
    logger.info("One-factor-at-a-time base setting:")
    for k, v in BASE_SIMSV2_HPARAMS.items():
        logger.info(f"  {k:<20}: {v}")

    summary_rows = load_existing_summary_rows(summary_path) if not init_args.no_resume else []
    finished_run_tags = load_finished_run_tags(summary_path) if not init_args.no_resume else set()

    if finished_run_tags:
        logger.info(f"Resume enabled: {len(finished_run_tags)} finished runs found in {summary_path}")

    for idx, setting in enumerate(sweep_plan, start=1):
        sweep_param = setting["sweep_param"]
        sweep_value = setting["sweep_value"]
        run_tag = f"seed{seed}_{sweep_param}_{sanitize_value(sweep_value)}"

        if run_tag in finished_run_tags:
            logger.info(f"\n[Skip {idx}/{len(sweep_plan)}] already finished: {run_tag}")
            continue

        logger.info("\n" + "=" * 90)
        logger.info(f"[{idx}/{len(sweep_plan)}] SIMS-v2 seed={seed} | {sweep_param}={sweep_value}")
        logger.info("=" * 90)

        # Load ConfigRegression every time, then explicitly restore main-table base hparams.
        config = ConfigRegression(init_args)
        args = config.get_config()
        args = copy_runtime_args(init_args, args)

        args.datasetName = DATASET_NAME
        args.train_mode = TRAIN_MODE
        args.seed = seed
        args.seeds = [seed]
        args.cur_time = 1

        apply_single_sweep(args, sweep_param, sweep_value)
        args.run_tag = run_tag

        # Put each run into its own output directory to avoid checkpoint/result overwriting.
        base_model_save_dir = init_args.model_save_dir
        base_res_save_dir = init_args.res_save_dir
        args.model_save_dir = os.path.join(base_model_save_dir, run_tag)
        args.res_save_dir = os.path.join(base_res_save_dir, run_tag)
        os.makedirs(args.model_save_dir, exist_ok=True)
        os.makedirs(args.res_save_dir, exist_ok=True)

        logger.info("Effective hyperparameters:")
        for key in [
            "cib_scale",
            "beta",
            "modality_dropout",
            "pseudo_tokens",
            "cnn_kernel",
            "cnn_stride",
            "qformer_layers",
            "subspace_dim",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
        ]:
            if hasattr(args, key):
                logger.info(f"  {key:<20}: {getattr(args, key)}")

        try:
            setup_seed(seed)
            results = run_one_setting(args)

            row = {
                "dataset": DATASET_NAME,
                "seed": seed,
                "sweep_param": sweep_param,
                "sweep_value": sweep_value,
                "run_tag": args.run_tag,
                "effective_cib_scale": getattr(args, "cib_scale", None),
                "effective_beta": getattr(args, "beta", None),
                "effective_modality_dropout": getattr(args, "modality_dropout", None),
                "effective_pseudo_tokens": getattr(args, "pseudo_tokens", None),
            }
            row.update(results)
            summary_rows.append(row)

            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            logger.info(f"[Saved] Intermediate summary: {summary_path}")

            metric_msg = []
            for k in ["MAE", "Corr", "F1_score", "Mult_acc_2", "Mult_acc_2_weak", "Mult_acc_3", "Mult_acc_5"]:
                if k in results:
                    try:
                        metric_msg.append(f"{k}={float(results[k]):.4f}")
                    except Exception:
                        metric_msg.append(f"{k}={results[k]}")
            logger.info("Result: " + " | ".join(metric_msg))

        except Exception as e:
            logger.error(f"Run failed for {sweep_param}={sweep_value}: {e}")
            import traceback

            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            row = {
                "dataset": DATASET_NAME,
                "seed": seed,
                "sweep_param": sweep_param,
                "sweep_value": sweep_value,
                "run_tag": f"FAILED_seed{seed}_{sweep_param}_{sanitize_value(sweep_value)}",
                "effective_cib_scale": BASE_SIMSV2_HPARAMS["cib_scale"],
                "effective_beta": BASE_SIMSV2_HPARAMS["beta"],
                "effective_modality_dropout": BASE_SIMSV2_HPARAMS["modality_dropout"],
                "effective_pseudo_tokens": BASE_SIMSV2_HPARAMS["pseudo_tokens"],
                "error": str(e),
            }
            summary_rows.append(row)
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    logger.info("\nAll sensitivity runs finished.")
    logger.info(f"Final summary saved to: {summary_path}")


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_tune", action="store_true", help="tune parameters?")
    parser.add_argument("--tune_mode", action="store_true", help="use valid set for tuning")
    parser.add_argument("--train_mode", type=str, default=TRAIN_MODE)
    parser.add_argument("--modelName", type=str, default="cmcm")
    parser.add_argument("--datasetName", type=str, default=DATASET_NAME)

    parser.add_argument("--root_dataset_dir", type=str, default="/home/xiewenbo/Dataset/multimodal_dataset/dataset")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model_save_dir", type=str, default="results/models_hparam_sensitivity_full")
    parser.add_argument("--res_save_dir", type=str, default="results/hparam_sensitivity_full")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--pretrain_LM", type=str, default="/home/xiewenbo/LLM/chatglm3-6b-base")
    parser.add_argument("--gpu_ids", type=int, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument(
        "--sweep_param",
        type=str,
        default="all",
        choices=["all", "beta", "cib_scale", "modality_dropout", "pseudo_tokens"],
        help="Which hyperparameter to sweep. Use 'all' for the full one-factor sensitivity set.",
    )
    parser.add_argument(
        "--sweep_values",
        type=str,
        default=None,
        help=(
            "Comma-separated custom values for a single sweep, e.g. "
            "--sweep_param cib_scale --sweep_values 0.05,0.1,0.15,0.2"
        ),
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resume. By default, existing successful run_tags in the summary CSV are skipped.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.datasetName = DATASET_NAME
    args.train_mode = TRAIN_MODE
    args.seed = int(args.seed)
    run_sensitivity(args)
