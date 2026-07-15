import os
import argparse

from utils.functions import Storage

class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'cmcm': self.__CMCM
        }
        # hyper parameters for datasets
        self.root_dataset_dir = args.root_dataset_dir
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        
        # integrate all parameters
        model_specific_params = HYPER_MODEL_MAP[model_name]()['datasetParas'].get(dataset_name, {})

        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **model_specific_params,
                            ))
    
    def __datasetCommonParams(self):
        root_dataset_dir = self.root_dataset_dir
        tmp = {
            'mosi':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    'feature_dims': (4096, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Has0_F1_score'
                }
            },
            'mosei':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    'feature_dims': (4096, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Has0_F1_score'
                }
            },
            'simsv2': {
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS_V2/ch-simsv2s.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (50, 925, 232),  
                    'feature_dims': (4096, 25, 177), 
                    'train_samples': 2722,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'F1_score',
                }
            }
        }
        return tmp

    def __CMCM(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_label_prefix': True,
                'need_normalized': False,
                'use_PLM': True,
                'save_labels': False,
                'cnn_kernel': 4,       
                

            },

            # dataset
            'datasetParas':{
                'mosei':{
                    'task_specific_prompt': 'Please predict the sentiment intensity of the above multimodal content in the range [-3.0, +3.0]. Assistant: The sentiment is',
                    'max_new_tokens': 4,
                    'batch_size': 16,
                    'learning_rate': 1e-4, 
                    'warm_up_epochs': 10,
                    'update_epochs': 1,
                    'early_stop': 8,
                    'H': 3.0,
                    'qformer_layers': 1, 
                    'lora_r': 8,                                
                    'lora_alpha': 16,                          
                    'lora_dropout': 0.3,                      
                    'lora_target_modules': ["query_key_value"],   

                    'pseudo_tokens': 16,      
                    'cnn_stride': 2,         
                    'subspace_dim': 128,     
                    'modality_dropout': 0.3,
                    'warmup_epochs_kl': 5,  
                    'target_kl': 0.005,
                    'cib_scale': 0.15,        
                    'beta': 1.2,            
                },

                'simsv2': {
                    #'task_specific_prompt': '请对上述多模态内容的情感强度进行预测，范围在[-1.0, +1.0]之间。响应: 情感为',
                    'task_specific_prompt': '请对上述多模态内容的情感强度进行预测，务必包含正负号(+或-)，范围在[-1.0, +1.0]之间。响应: 情感为',
                    'max_new_tokens': 4,
                    'batch_size': 16,
                    'learning_rate': 1e-4,
                    'warm_up_epochs': 5,
                    'update_epochs': 1,
                    'early_stop': 8,
                    'H': 1.0,
                    'qformer_layers': 1,   

                    'lora_r': 4,                                
                    'lora_alpha': 8,                          
                    'lora_dropout': 0.4,                      
                    'lora_target_modules': ["query_key_value"],   

                    'pseudo_tokens': 8,
                    'cnn_stride': 2,        
                    'subspace_dim': 128,
                    'modality_dropout': 0.3,
                    'warmup_epochs_kl':0,  
                    'target_kl': 0.005,      
                    'cib_scale': 0.15,       
                    'beta':1.2,
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args