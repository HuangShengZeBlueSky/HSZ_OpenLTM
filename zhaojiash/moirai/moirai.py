# ============================================
#       HuggingFace Moirai 微调脚本（修正参数问题）
# ============================================

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import joblib
import os
import json
from safetensors.torch import load_file
import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from uni2ts.model.moirai import MoiraiModule,MoiraiFinetune
from uni2ts.loss.packed import PackedNLLLoss
from data_provider.data_loader import BJTUAnomalyloader
from uni2ts.distribution import StudentTOutput

# ===============================
#           数据配置
# ===============================
CONTEXT = 496
PRED = 16
PATCH_SIZE = 16
BATCH = 64
EPOCHS = 10
LR = 1e-5


PRETRAINED_DIR = "/home/ubuntu/zhaojia/checkpoints/Moirai"
SAVE_DIR = "/home/ubuntu/zhaojia/checkpoints/moirai-finetuned"
os.makedirs(SAVE_DIR, exist_ok=True)


class MoiraiFinetuneDataset(Dataset):
    def __init__(self, bjtu_loader, context_len, pred_len, patch_size):
        self.bjtu_loader = bjtu_loader
        self.context_len = context_len
        self.pred_len = pred_len
        self.total_len = context_len + pred_len
        self.patch_size = patch_size

        # 读取变量维度 C
        sample_x, _, _, _ = bjtu_loader[0]
        self.variate_dim = sample_x.shape[1]
        print(f"[Dataset] variate_dim = {self.variate_dim}")

    def __len__(self):
        return len(self.bjtu_loader)
    def __getitem__(self, idx):
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.bjtu_loader[idx]
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        L_orig, C_orig = seq_x.shape

        # 强制统一时间序列长度 (L) 和变量维度 (C)
        L_target = self.total_len  # CONTEXT + PRED
        C_target = self.variate_dim

        # 时间维度处理
        if L_orig < L_target:
            pad = torch.zeros(L_target - L_orig, C_orig)
            seq_x = torch.cat([seq_x, pad], dim=0)
        elif L_orig > L_target:
            seq_x = seq_x[:L_target]

        # 变量维度处理
        if C_orig < C_target:
            pad = torch.zeros(L_target, C_target - C_orig)
            seq_x = torch.cat([seq_x, pad], dim=1)
        elif C_orig > C_target:
            seq_x = seq_x[:, :C_target]

        L, C = L_target, C_target
        
        # 关键修改：添加 patch 维度 (max_patch=1，或根据实际情况调整)
        target = seq_x.unsqueeze(-1)  # 形状: (L, C, 1)
        
        # observed_mask 形状: (L, C, 1)
        observed_mask = torch.ones(L, C, 1, dtype=torch.bool)
        
        # prediction_mask 形状: (L, C)，后续会被 unsqueeze(-1) 为 (L, C, 1)
        prediction_mask = torch.zeros(L, C, dtype=torch.bool)
        prediction_mask[self.context_len:, :] = True  # 只预测未来部分

        sample_id = torch.full((L, C), idx, dtype=torch.long)
        time_id = torch.arange(L).unsqueeze(1).repeat(1, C)  # (L, C)
        variate_id = torch.arange(C).unsqueeze(0).repeat(L, 1)  # (L, C)
        patch_id = (time_id // self.patch_size).long()  # (L, C)

        return {
            "target": target,
            "observed_mask": observed_mask,
            "prediction_mask": prediction_mask,
            "sample_id": sample_id,
            "time_id": time_id,
            "variate_id": variate_id,
            "patch_id": patch_id,
            "patch_size": torch.tensor(self.patch_size),
        }
def collate_fn(batch):
    out = {}
    for k in batch[0]:
        if k == "patch_size":
            # patch_size 是全局超参数，所有样本相同，取第一个即可（保持为标量）
            out[k] = batch[0][k]
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
# ===============================
#           数据模块
# ===============================
class MoiraiDataModule(L.LightningDataModule):
    def __init__(self, train_loader, valid_loader, context_len, pred_len, patch_size, batch_size):
        super().__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.context_len = context_len
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = MoiraiFinetuneDataset(
            self.train_loader, self.context_len, self.pred_len, self.patch_size
        )
        self.valid_dataset = MoiraiFinetuneDataset(
            self.valid_loader, self.context_len, self.pred_len, self.patch_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            drop_last=False
        )

# ===============================
#           训练函数
# ===============================
def main():
    # --- 加载模型配置 ---
    config_path = os.path.join(PRETRAINED_DIR, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # 提取原始 patch_size（可能是 int）
    raw_patch_size = config_dict.get('patch_size', PATCH_SIZE)

    # 构建 filtered_config，但把 patch_size 改为 patch_sizes（列表）
    filtered_config = {}
    for k, v in config_dict.items():
        if k in ['attn_dropout_p', 'd_model', 'dropout_p', 'max_seq_len', 
                'num_layers', 'quantile_levels', 'scaling']:
            filtered_config[k] = v

    # 特别处理 patch_size → patch_sizes
    filtered_config['patch_sizes'] = [raw_patch_size]  # ← 关键修改！
    filtered_config['distr_output'] = StudentTOutput() 


    
    # --- 加载预训练模型 ---
    print("Loading pretrained Moirai model...")
    try:
        # 创建基础模块
        module = MoiraiModule(**filtered_config)
        
        # 加载safetensors权重
        model_path = os.path.join(PRETRAINED_DIR, "model.safetensors")
        if os.path.exists(model_path):
            state_dict = load_file(model_path, device="cpu")
            missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            print("Loaded pretrained weights successfully")
        else:
            print("No model.safetensors found, training from scratch")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        # 如果还有错误，尝试使用最少的必需参数
        print("Trying with minimal parameters...")
        minimal_config = {
            'd_model': config_dict['d_model'],
            'num_layers': config_dict['num_layers'],
            'patch_sizes': [config_dict['patch_size']],  # ✅ 包装成列表
            'max_seq_len': config_dict['max_seq_len'],
            'attn_dropout_p': config_dict.get('attn_dropout_p', 0.0),
            'dropout_p': config_dict.get('dropout_p', 0.0),
            'distr_output': StudentTOutput(), 
        }
        module = MoiraiModule(**minimal_config)
    
    # --- 加载数据 ---
    print("Loading BJTU data...")
    train_loader = BJTUAnomalyloader(
        root_path="/home/ubuntu/zhaojia/OpenLTM_data_backup/datasets/中车研究院实验台数据集/工况1_转速12转每秒/train/正常运行0-1.1_170623135209.csv",
        seq_len=CONTEXT + PRED,
        patch_len=PATCH_SIZE,
        flag="train",
    )
    
    valid_loader = BJTUAnomalyloader(
        root_path="/home/ubuntu/zhaojia/OpenLTM_data_backup/datasets/中车研究院实验台数据集/工况1_转速12转每秒/train/正常运行0-1.1_170623135209.csv",
        seq_len=CONTEXT + PRED,
        patch_len=PATCH_SIZE,
        flag="valid",
        scaler=train_loader.scaler
    )
    
    print(f"Train samples: {len(train_loader)}, Valid samples: {len(valid_loader)}")
    
    # 计算训练步数
    num_training_steps = (len(train_loader) // BATCH) * EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    
    print(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

    # --- 创建MoiraiFinetune模块 ---
    print("Creating MoiraiFinetune module...")
    
    # 配置微调参数
    finetune_kwargs = {
        "min_patches": 1,
        "min_mask_ratio": 0.0,
        "max_mask_ratio": 0.0,  # 对于微调，可以禁用masking
        "max_dim": 2,  # 你的数据是2维的
        "num_training_steps": num_training_steps,
        "num_warmup_steps": num_warmup_steps,
        "module": module,
        "lr": LR,
        "weight_decay": 1e-2,
        "context_length": CONTEXT,
        "prediction_length": PRED,
        "patch_size": PATCH_SIZE,
        "finetune_pattern": "full",  # 全参数微调
        "loss_func": PackedNLLLoss(),
        "log_on_step": True,
    }
    
    model = MoiraiFinetune(**finetune_kwargs)
    
    # --- 创建数据模块 ---
    datamodule = MoiraiDataModule(
        train_loader, valid_loader, CONTEXT, PRED, PATCH_SIZE, BATCH
    )

    # --- 设置回调函数 ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=SAVE_DIR,
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val/PackedNLLLoss",
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/PackedNLLLoss",
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode="min"
    )
    
    # --- 创建训练器 ---
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=CSVLogger(SAVE_DIR),
        accelerator="gpu",
        devices=1,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
    )

    # --- 开始训练 ---
    print("Starting fine-tuning...")
    trainer.fit(model, datamodule=datamodule)

    # --- 保存最终模型 ---
    print("Saving final model...")
    
    # 保存最佳模型
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Best model saved at: {best_model_path}")
    
    # 保存模块状态
    torch.save(model.module.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))
    
    # 保存配置
    with open(os.path.join(SAVE_DIR, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # 可选：保存scaler
    if hasattr(train_loader, 'scaler') and train_loader.scaler is not None:
        joblib.dump(train_loader.scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
        print("Scaler saved")
    else:
        print("No scaler found, skipping scaler save")
    
    print("Fine-tuning completed!")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()