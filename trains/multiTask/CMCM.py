import os
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import matplotlib
from itertools import chain

logger = logging.getLogger('MSA')

class CMCM():
    def __init__(self, args):
        self.args = args
        self.args.tasks = "M"
        self.metrics = MetricsTop(args).getMetics(args.datasetName)

        self.feature_map = {
            'fusion': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device),
        }

        self.dim_map = {
            'fusion': torch.tensor(args.post_fusion_dim).float(),
            'text': torch.tensor(args.post_text_dim).float(),
            'audio': torch.tensor(args.post_audio_dim).float(),
            'vision': torch.tensor(args.post_video_dim).float(),
        }
        
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }

    def do_train(self, model, dataloader):
        scaler = GradScaler()
        
        optimizer = optim.AdamW(model.parameters(), lr=self.args.learning_rate, eps=1e-4)
        
        total_steps = len(dataloader['train']) * self.args.warm_up_epochs
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)
        
        saved_labels = {}
        logger.info("Init labels...")
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        losses = []

        min_or_max = 'min' if self.args.KeyEval in ['MAE'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0 
        
        while True: 
            epochs += 1
            y_pred = {'M': []}
            y_true = {'M': []}
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            
            with tqdm(enumerate(dataloader['train']), total=len(dataloader['train'])) as td:
                for i, batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()      
                    left_epochs -= 1                

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    
                    if self.args.train_mode == 'regression':
                        labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                    else:
                        labels_m = batch_data['labels']['M'].to(self.args.device)

                    if not self.args.need_data_aligned:
                        text_lengths = batch_data['text_lengths'].to(self.args.device)
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)

                    warmup_epochs_kl = getattr(self.args, 'warmup_epochs_kl', 10)
                    target_kl = getattr(self.args, 'target_kl', 0.005)
                    steps_per_epoch = len(dataloader['train'])
                    total_warmup_steps = steps_per_epoch * warmup_epochs_kl
                    current_global_step = (epochs - 1) * steps_per_epoch + i
                    
                    if current_global_step < total_warmup_steps:
                        cur_kl = target_kl * (current_global_step / total_warmup_steps)
                    else:
                        cur_kl = target_kl

                    with autocast():
                        output = model(
                            labels_m, 
                            (text, text_lengths), 
                            (audio, audio_lengths), 
                            (vision, vision_lengths), 
                            cur_kl_weight=cur_kl
                        )
                        loss = output['Loss']

                    scaler.scale(loss).backward()
                    train_loss += loss.item()
                    
                    if not left_epochs:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        left_epochs = self.args.update_epochs
                
                if not left_epochs:
                    scaler.step(optimizer)
                    scaler.update()
            
            train_loss = train_loss / len(dataloader['train'])

            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f KL_Weight: %.5f" % (self.args.modelName, \
                        epochs-best_epoch, epochs, self.args.cur_time, train_loss, cur_kl))
            losses.append(train_loss)

            # validation
            if epochs >= 1:
                val_results = self.do_test(model, dataloader['valid'], mode="VAL")
                cur_valid = val_results[self.args.KeyEval]
                
                isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
                if isBetter:
                    best_valid, best_epoch = cur_valid, epochs
                    self.save_model(model, epochs, self.args.model_save_path)
                    model.to(self.args.device)

                if epochs - best_epoch >= self.args.early_stop:
                    if self.args.save_labels:
                        with open(os.path.join(self.args.res_save_dir, f'{self.args.modelName}-{self.args.datasetName}-labels.pkl'), 'wb') as df:
                            plk.dump(saved_labels, df, protocol=4)
                    return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': []}
        y_true = {'M': []}
        
        if self.args.train_mode == 'regression':
            with torch.no_grad():
                with tqdm(dataloader) as td:
                    for batch_data in td:
                        vision = batch_data['vision'].to(self.args.device)
                        audio = batch_data['audio'].to(self.args.device)
                        text = batch_data['text'].to(self.args.device)
                        if not self.args.need_data_aligned:
                            text_lengths = batch_data['text_lengths'].to(self.args.device)
                            audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                            vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                        
                        with autocast():
                            outputs = model.generate((text,text_lengths), (audio, audio_lengths), (vision, vision_lengths))

                        predict_label = torch.tensor(outputs).to(self.args.device)
                        labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                        y_pred['M'].append(predict_label.cpu())
                        y_true['M'].append(labels_m.cpu())
            
            # 转为 numpy 方便处理
            pred = torch.cat(y_pred['M'])
            true = torch.cat(y_true['M'])

            
            p_arr = np.array(pred)
            t_arr = np.array(true)
            
            # 计算指标
            eval_results = self.metrics(pred, true)

            logger.info(mode + "-(%s)" % self.args.modelName + " >>" )
            logger.info('M: >> ' + dict_to_str(eval_results))
            
        else:
            # Classification Logic
            with torch.no_grad():
                with tqdm(dataloader) as td:
                    for batch_data in td:
                        vision = batch_data['vision'].to(self.args.device)
                        audio = batch_data['audio'].to(self.args.device)
                        text = batch_data['text'].to(self.args.device)
                        if not self.args.need_data_aligned:
                            text_lengths = batch_data['text_lengths'].to(self.args.device)
                            audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                            vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                        with autocast():
                            outputs = model.generate((text, text_lengths), (audio, audio_lengths),
                                                     (vision, vision_lengths))

                        if isinstance(outputs, torch.Tensor) and outputs.dim() > 1:
                            predict_label = torch.argmax(outputs, dim=-1)
                        else:
                            predict_label = outputs
                        
                        labels_m = batch_data['labels']['M']
                        
                        if isinstance(predict_label, torch.Tensor):
                            predict_label = predict_label.cpu()
                        if isinstance(labels_m, torch.Tensor):
                            labels_m = labels_m.cpu()
                            
                        y_pred['M'].append(predict_label)
                        y_true['M'].append(labels_m)
            
            pred = list(chain(*y_pred['M']))
            true = list(chain(*y_true['M']))
            
            # --- 新增：打印前20个预测值与真实值 ---
            # 如果 pred/true 内部元素是 tensor，转换为标量方便查看
            pred_print = [p.item() if isinstance(p, torch.Tensor) else p for p in pred[:20]]
            true_print = [t.item() if isinstance(t, torch.Tensor) else t for t in true[:20]]
            logger.info(f"{mode} -(Classification) 前20个预测值: {pred_print}")
            logger.info(f"{mode} -(Classification) 前20个真实值: {true_print}")
            # -------------------------------------
            
            eval_results = self.metrics(pred, true)
            logger.info(mode + "-(%s)" % self.args.modelName + " >>")
            logger.info('M: >> ' + dict_to_str(eval_results))

        return eval_results

    def save_model(self, model, epoch, save_path):
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model.named_parameters()
        }
        state_dict = model.cpu().state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                del state_dict[k]
        logging.info("Saving checkpoint at epoch {} to {}.".format(epoch, save_path))
        torch.save(state_dict, save_path)