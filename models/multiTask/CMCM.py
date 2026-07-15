import math
import os
import sys
import collections
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.Textmodel import Language_model

__all__ = ['CMCM']

class UGLoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, lora_alpha=16, lora_dropout=0.05, uncertainty_dim=256):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

        in_features = base_layer.weight.shape[1]
        out_features = base_layer.weight.shape[0]

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        self.uncertainty_router = nn.Sequential(
            nn.Linear(uncertainty_dim, r),
            nn.SiLU(),
            nn.Linear(r, r),
            nn.Sigmoid() 
        )

        self.dropout = nn.Dropout(lora_dropout)
        
        self.muc = None 

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        nn.init.normal_(self.uncertainty_router[0].weight, std=0.02)
        nn.init.normal_(self.uncertainty_router[2].weight, std=0.02)

        self.base_layer.weight.requires_grad = False
        if getattr(self.base_layer, 'bias', None) is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)

        if self.r > 0:
            lora_out = self.lora_A(self.dropout(x)) 

            if self.muc is not None:
                lambda_r = self.uncertainty_router(self.muc) 
                
                if lora_out.shape[0] == lambda_r.shape[0]:
                    lora_out = lora_out * lambda_r.unsqueeze(1)
                else:
                    lora_out = lora_out * lambda_r.unsqueeze(0)

            lora_out = self.lora_B(lora_out) * self.scaling
            result = result + lora_out

        return result

def inject_ug_lora(model, target_module_names=["query_key_value"], r=8, alpha=16, dropout=0.05, uncertainty_dim=256):
    for name, module in model.named_children():
        if any(target_key in name for target_key in target_module_names):
            ug_layer = UGLoRALinear(module, r=r, lora_alpha=alpha, lora_dropout=dropout, uncertainty_dim=uncertainty_dim)
            setattr(model, name, ug_layer)
        else:
            inject_ug_lora(module, target_module_names, r, alpha, dropout, uncertainty_dim)

def set_ug_context(model, muc_feat):
    for module in model.modules():
        if isinstance(module, UGLoRALinear):
            module.muc = muc_feat


class CMCM(nn.Module):
    def __init__(self, args):
        super(CMCM, self).__init__()
        self.LLM = Language_model(args)

        text_in, audio_in, video_in = args.feature_dims[:]

        if text_in != 4096:
            print(f"[Warning] Config text_in is {text_in}, but ChatGLM3 requires 4096. Adjusting target projection...")
            target_llm_dim = 4096
        else:
            target_llm_dim = text_in

        num_classes = getattr(args, 'num_classes', 3)
        self.hidden_dim = 256 

        self.cnn_kernel = getattr(args, 'cnn_kernel', 3) 
        self.cnn_stride = getattr(args, 'cnn_stride', 2)
        self.qformer_layers = getattr(args, 'qformer_layers', 1)
        
        self.subspace_dim = getattr(args, 'subspace_dim', 128)
        self.dropout_rate = getattr(args, 'modality_dropout', 0.3)
        
        self.beta = getattr(args, 'beta', 0.5) 
        self.cib_scale = getattr(args, 'cib_scale', 0.1)

        self.audio_proj = CnnProjector(audio_in, self.hidden_dim, kernel_size=self.cnn_kernel, stride=self.cnn_stride)
        self.video_proj = CnnProjector(video_in, self.hidden_dim, kernel_size=self.cnn_kernel, stride=self.cnn_stride)
        self.text_proj_layer = nn.Linear(target_llm_dim, self.hidden_dim)

        self.audio_dits = DITS_Module_Seq(self.hidden_dim, self.subspace_dim, num_classes=num_classes, dropout_rate=self.dropout_rate)
        self.video_dits = DITS_Module_Seq(self.hidden_dim, self.subspace_dim, num_classes=num_classes, dropout_rate=self.dropout_rate)
        self.post_dits_norm = nn.LayerNorm(self.hidden_dim)
        
        self.qformer_fusion = QFormerFusion(
            input_dim=self.hidden_dim, 
            output_dim=target_llm_dim,  
            num_queries=args.pseudo_tokens, 
            num_layers=self.qformer_layers,
            dropout=self.dropout_rate
        )

        raw_rank = getattr(args, 'lora_rank', 8)
        lora_rank = 8 if type(raw_rank) is bool else raw_rank
        
        raw_alpha = getattr(args, 'lora_alpha', 16)
        lora_alpha = 16 if type(raw_alpha) is bool else raw_alpha
        
        inject_ug_lora(
            self.LLM.model, 
            target_module_names=["query_key_value"], 
            r=lora_rank, 
            alpha=lora_alpha, 
            dropout=0.1,  # 适度加大 Dropout 防过拟合
            uncertainty_dim=self.subspace_dim * 2 # 音频 + 视频的 logvar 拼接维度
        )
        
        self.LLM.model.gradient_checkpointing_enable()

        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
        
    def forward(self, labels, text, audio, video, cur_kl_weight=0.0):
        audio_feat = audio[0] if isinstance(audio, (list, tuple)) else audio
        video_feat = video[0] if isinstance(video, (list, tuple)) else video
        text_ids = text[0] if isinstance(text, (list, tuple)) else text
        
        if text_ids.dim() == 3:
            input_ids = text_ids[:, 0, :].long()
        else:
            input_ids = text_ids.long()
            
        text_emb = self.LLM.text_embedding(input_ids)
        
        audio_h = self.audio_proj(audio_feat)
        video_h = self.video_proj(video_feat)
        text_h = self.text_proj_layer(text_emb)

        audio_steered, a_cib_loss, a_aux_loss, a_kl_loss, a_logvar = self.audio_dits(
            audio_h, text_h, labels=labels, is_training=self.training, kl_weight=cur_kl_weight, beta=self.beta
        )
        video_steered, v_cib_loss, v_aux_loss, v_kl_loss, v_logvar = self.video_dits(
            video_h, text_h, labels=labels, is_training=self.training, kl_weight=cur_kl_weight, beta=self.beta
        )

        audio_steered = self.post_dits_norm(audio_steered)
        video_steered = self.post_dits_norm(video_steered)

        fusion_tokens = self.qformer_fusion(audio_steered, video_steered, text_h)

        muc = torch.cat([a_logvar, v_logvar], dim=-1)
        set_ug_context(self.LLM.model, muc.detach())
        
        LLM_input = torch.cat([fusion_tokens, text_emb], dim=1)
        LLM_output = self.LLM(LLM_input, labels)

        final_loss = LLM_output.loss + self.cib_scale * (a_cib_loss + v_cib_loss)

        res = {
            'Loss': final_loss,
            'LLM_Loss': LLM_output.loss,
            'KL_Loss': a_kl_loss + v_kl_loss,
            'Aux_Acc_A': 1.0/ (a_aux_loss + 1e-5), 
            'Feature_f': fusion_tokens,
        }
        return res

    def generate(self, text, audio, video):
        audio_feat = audio[0] if isinstance(audio, (list, tuple)) else audio
        video_feat = video[0] if isinstance(video, (list, tuple)) else video
        text_ids = text[0] if isinstance(text, (list, tuple)) else text
        
        if text_ids.dim() == 3:
            input_ids = text_ids[:, 0, :].long()
        else:
            input_ids = text_ids.long()

        text_emb = self.LLM.text_embedding(input_ids)
        
        audio_h = self.audio_proj(audio_feat)
        video_h = self.video_proj(video_feat)
        text_h = self.text_proj_layer(text_emb)

        audio_steered, _, _, _, a_logvar = self.audio_dits(audio_h, text_h, is_training=False)
        video_steered, _, _, _, v_logvar = self.video_dits(video_h, text_h, is_training=False)

        audio_steered = self.post_dits_norm(audio_steered)
        video_steered = self.post_dits_norm(video_steered)

        fusion_tokens = self.qformer_fusion(audio_steered, video_steered, text_h)

        # 🌟 路由通知
        muc = torch.cat([a_logvar, v_logvar], dim=-1)
        set_ug_context(self.LLM.model, muc)

        LLM_input = torch.cat([fusion_tokens, text_emb], dim=1)
        LLM_output = self.LLM.generate(LLM_input)

        return LLM_output


class CnnProjector(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=2, dropout=0.2):
        super(CnnProjector, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
            nn.BatchNorm1d(out_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv_net(x)
        out = out.transpose(1, 2)
        return out

class DITS_Module_Seq(nn.Module):
    def __init__(self, input_dim, subspace_dim=128, num_classes=3, dropout_rate=0.3):
        super(DITS_Module_Seq, self).__init__()
        self.num_classes = num_classes  
        
        self.text_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, dropout=dropout_rate, batch_first=True)

        self.down_proj_v = nn.Linear(input_dim, subspace_dim)
        self.down_proj_t = nn.Linear(input_dim, subspace_dim)
        self.up_proj = nn.Linear(subspace_dim, input_dim)
        
        self.gate_net = nn.Sequential(
            nn.Linear(subspace_dim * 2, subspace_dim),
            nn.ReLU(),
            nn.Linear(subspace_dim, 1),
            nn.Sigmoid()
        )
        
        self.logvar_layer = nn.Linear(subspace_dim * 2, subspace_dim)

        self.aux_classifier = nn.Sequential(
            nn.Linear(subspace_dim + input_dim, subspace_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(subspace_dim, num_classes) 
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        for m in [self.down_proj_v, self.down_proj_t, self.aux_classifier[0], self.aux_classifier[2]]:
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)
        
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        nn.init.constant_(self.logvar_layer.weight, 0)
        nn.init.constant_(self.logvar_layer.bias, -10.0)

    def forward(self, v_feat, t_feat, labels=None, is_training=True, kl_weight=0.0, beta=0.5):
        t_aligned, _ = self.text_attn(query=v_feat, key=t_feat, value=t_feat)
        
        z_v = self.down_proj_v(v_feat)
        z_t = self.down_proj_t(t_aligned) 
        
        concat_feat = torch.cat([z_v, z_t], dim=-1)
        gate = self.gate_net(concat_feat)
        mu = z_v * gate 
        
        concat_feat_pooled = concat_feat.mean(dim=1)
        logvar = self.logvar_layer(concat_feat_pooled)
        
        cib_loss = torch.tensor(0.0, device=v_feat.device)
        aux_loss = torch.tensor(0.0, device=v_feat.device)
        kl_loss = torch.tensor(0.0, device=v_feat.device)
        
        if is_training and labels is not None:
            mu_pooled = mu.mean(dim=1)
            
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_sample = mu_pooled + eps * std 
            
            t_feat_pooled = t_feat.mean(dim=1)
            aux_input = torch.cat([z_sample, t_feat_pooled], dim=-1) 
            pred_logits = self.aux_classifier(aux_input)
            
            labels = labels.to(pred_logits.device)
            
            if self.num_classes == 3 and (labels.dtype == torch.float or labels.dtype == torch.float16):
                aux_labels = torch.ones_like(labels).long()
                aux_labels[labels > 0.1] = 2 
                aux_labels[labels < -0.1] = 0 
                aux_loss = F.cross_entropy(pred_logits, aux_labels.view(-1))
            else:
                aux_loss = F.cross_entropy(pred_logits, labels.long().view(-1))
            
            kl = -0.5 * torch.sum(1 + logvar - mu_pooled.pow(2) - logvar.exp(), dim=-1)
            kl_loss = kl.mean()
            cib_loss = kl_weight * kl_loss + beta * aux_loss

        delta_v = self.up_proj(self.dropout(mu))
        v_out = v_feat + delta_v
        
        return v_out, cib_loss, aux_loss, kl_loss, logvar 

class QFormerFusion(nn.Module):
    def __init__(self, input_dim=256, output_dim=4096, num_queries=4, num_layers=1, num_heads=4, dropout=0.2):
        super(QFormerFusion, self).__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, input_dim))
        self.modality_type_embeddings = nn.Embedding(2, input_dim)
        self.modality_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.layers = nn.ModuleList([
            QFormerBlock(input_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(input_dim)
        self.output_proj = nn.Linear(input_dim, output_dim)

    def forward(self, audio_seq, video_seq, text_seq):
        B = audio_seq.shape[0]
        dev = audio_seq.device
        type_a = self.modality_type_embeddings(torch.tensor(0, device=dev))
        type_v = self.modality_type_embeddings(torch.tensor(1, device=dev))
        audio_seq = audio_seq + type_a
        video_seq = video_seq + type_v
        context = torch.cat([audio_seq, video_seq], dim=1)
        queries = self.query_tokens.repeat(B, 1, 1)
        for layer in self.layers:
            queries = layer(queries, context, text_seq)
        out = self.norm(queries)
        return self.output_proj(out)

class QFormerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.2):
        super(QFormerBlock, self).__init__()
        self.text_cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mm_cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, queries, context, text_seq):
        text_out, _ = self.text_cross_attn(query=queries, key=text_seq, value=text_seq)
        queries = self.norm1(queries + text_out)
        mm_out, _ = self.mm_cross_attn(query=queries, key=context, value=context)
        queries = self.norm2(queries + mm_out)
        ffn_out = self.ffn(queries)
        queries = self.norm3(queries + ffn_out)
        return queries