import os
import gc
import time
import random
import torch
import pynvml
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy 

# 假设你的文件结构没变，直接引用
from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from config.config_regression import ConfigRegression
from config.config_classification import ConfigClassification

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# [关键修改 1] 强制关闭 Tokenizer 并行，防止 DataLoader 卡死
# ==========================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # [优化] LLM 微调建议关闭 benchmark 以获得确定的显存占用

def set_log(args):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_file_path = f'logs/{args.modelName}-{args.datasetName}.log'
    
    global logger
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger

def run(args):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(args.model_save_dir, \
                                        f'{args.modelName}-{args.datasetName}-{args.train_mode}.pth')
    
    # 自动显存选择逻辑
    if len(args.gpu_ids) == 0 and torch.cuda.is_available():
        try:
            pynvml.nvmlInit()
            dst_gpu_id, min_mem_used = 0, 1e16
            # 扫描前两张卡，或者你可以改为 range(torch.cuda.device_count())
            for g_id in [6]: 
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used = meminfo.used
                    if mem_used < min_mem_used:
                        min_mem_used = mem_used
                        dst_gpu_id = g_id
                except:
                    continue
            args.gpu_ids.append(dst_gpu_id)
            print(f'Find gpu: {dst_gpu_id}, use memory: {min_mem_used}!')
        except Exception as e:
            print(f"GPU auto-select failed: {e}. Defaulting to 0.")
            args.gpu_ids.append(0)

    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device
    
    # data
    dataloader = MMDataLoader(args)
    
    # Init Model (AMIO 需要确保引用了新的 ChatGLMModel)
    model = AMIO(args).to(device)
    
    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        logger.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    print_trainable_parameters(model)

    # Init Trainer
    atio = ATIO().getTrain(args)
    
    # Do Train
    atio.do_train(model, dataloader)
    
    # Load Best Model for Testing
    if os.path.exists(args.model_save_path):
        # 注意：如果你的 save_model 用的是 save_pretrained，这里要改用 from_pretrained
        # 假设你按照我给的 Trainer 用了 state_dict 保存：
        checkpoint = torch.load(args.model_save_path)
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)

    # Do Test
    if args.tune_mode:
        results = atio.do_test(model, dataloader['valid'], mode="VALID")
    else:
        results = atio.do_test(model, dataloader['test'], mode="TEST")

    # ==========================================
    # [关键修改 2] 彻底的显存清理
    # ==========================================
    del model
    del atio
    del dataloader
    torch.cuda.empty_cache()
    gc.collect()

    return results

def run_normal(args):
    global logger
    if not logger.handlers:
        set_log(args)

    args.res_save_dir = os.path.join(args.res_save_dir)
    init_args = args 
    seeds = args.seeds
    
    acc_records = [] 

    for i, seed in enumerate(seeds):
        args = copy.deepcopy(init_args)
        
        # Load Config
        if args.train_mode == "regression":
            config = ConfigRegression(args)
        else :
            config = ConfigClassification(args)
        
        args = config.get_config()

        # Hyperparameter Override Logic
        override_keys = [
            'beta', 'cib_scale', 'modality_dropout', 
            'cnn_kernel', 'cnn_stride', 
            'qformer_layers', 'pseudo_tokens', 
            'subspace_dim'
        ]
        
        for key in override_keys:
            if hasattr(init_args, key) and getattr(init_args, key) is not None:
                new_val = getattr(init_args, key)
                setattr(args, key, new_val)
                if i == 0:
                    print(f"  [Override Hyperparam] {key}: {new_val}")

        setup_seed(seed)
        args.seed = seed
        
        args.cur_time = i + 1
        
        # Robust Argument Printing
        logger.info("\n" + "="*30 + f" Configuration for Seed {seed} " + "="*30)
        
        args_dict = {}
        if isinstance(args, dict):
            args_dict = args
        elif hasattr(args, '__dict__'):
            args_dict = vars(args)
        
        if args_dict:
            try:
                for key in sorted(args_dict.keys()):
                    val = args_dict[key]
                    logger.info(f"{key:<25}: {val}")
            except Exception as e:
                logger.info(f"Print Error: {e}")
                logger.info(str(args))
        else:
            logger.info("Object has no __dict__ or is empty, printing raw object:")
            logger.info(str(args))
            
        logger.info("="*85 + "\n")

        try:
            test_results = run(args)
            
            # Record Accuracy/Results
            # 假设分类任务用 'Mult_acc_2'，回归用 'MAE'，根据你的 metrics 修改
            metric_key = 'Mult_acc_2' if args.train_mode == 'classification' else 'MAE'
            if metric_key in test_results:
                acc_records.append(test_results[metric_key])
                print(f"  > Seed {seed} {metric_key}: {test_results[metric_key]:.4f}")

            criterions = list(test_results.keys())
            save_path = os.path.join(args.res_save_dir, f'{args.datasetName}-{args.train_mode}.csv')
            if not os.path.exists(args.res_save_dir):
                os.makedirs(args.res_save_dir)
            
            if os.path.exists(save_path):
                df = pd.read_csv(save_path)
            else:
                df = pd.DataFrame(columns=["Model", "Seed"] + criterions)

            res_row = [args.modelName, f'{seed}']
            for c in criterions:
                res_row.append(round(test_results[c] * 100, 2))

            # 简单的列数对齐检查
            if len(res_row) != len(df.columns):
                 # 如果列不对齐，这里可以做额外处理，或者让 pandas 报错
                 pass 

            df.loc[len(df)] = res_row
            df.to_csv(save_path, index=None)
            logger.info('Results are added to %s...' % (save_path))

        except Exception as e:
            print(f"Error in Seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            # 出错后也尝试清理显存
            torch.cuda.empty_cache()
            return 0.0

    if len(acc_records) > 0:
        return np.mean(acc_records)
    else:
        return 0.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_tune', type=bool, default=False, help='tune parameters ?')
    parser.add_argument('--tune_mode', type=bool, default=False, help='use valid set for tuning')
    parser.add_argument('--train_mode', type=str, default="regression", help='regression / classification')
    parser.add_argument('--modelName', type=str, default='cmcm', help='support CMCM')
    parser.add_argument('--datasetName', type=str, default='sims', help='support mosi/mosei/simsv2/iemocap/meld/cherma')
    
    # Path Arguments
    parser.add_argument('--root_dataset_dir', type=str, default='/home/oydq/dataset/Dateset', help='Location of the root directory')
    parser.add_argument('--num_workers', type=int, default=0, help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/models', help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/results', help='path to save results.')
    
    parser.add_argument('--pretrain_LM', type=str, default='/home/oydq/chatglm3-6b-base', help='path to load pretrain LLM.')
    
    parser.add_argument('--gpu_ids', type=list, default=[], help='indicates the gpus will be used')
    
    # Hyperparameters Override (用于命令行调参)
    parser.add_argument('--beta', type=float, default=None, help='Aux Loss Weight')
    parser.add_argument('--cib_scale', type=float, default=None, help='DITS Module Weight')
    parser.add_argument('--modality_dropout', type=float, default=None, help='Modality Dropout Prob')

    parser.add_argument('--cnn_kernel', type=int, default=None, help='CNN kernel size')
    parser.add_argument('--cnn_stride', type=int, default=None, help='CNN stride size')
    parser.add_argument('--qformer_layers', type=int, default=None, help='Num of Q-Former layers')
    parser.add_argument('--pseudo_tokens', type=int, default=None, help='Num of Q-Former tokens')      
    parser.add_argument('--subspace_dim', type=int, default=None, help='DITS subspace dimension')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    set_log(args) 
    
    for data_name in ['cherma','simsv2', 'mosei' , 'meld']:
    #for data_name in ['cherma']:
        if data_name in ['mosi', 'mosei', 'sims', 'simsv2']:
            args.train_mode = 'regression'
        else:
            args.train_mode = 'classification'

        args.datasetName = data_name
        args.seeds = [1111, 2222, 3333, 4444, 5555]
        #args.seeds = [1111] 
        
        final_acc = run_normal(args)
        print(f"Final Mean Metric: {final_acc}")