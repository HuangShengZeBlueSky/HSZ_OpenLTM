#!/usr/bin/env python
# coding: utf-8

import os
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from models import timer_xl

# 创建结果保存目录
os.makedirs("results", exist_ok=True)

# 初始化模型参数
args = argparse.Namespace()
args.input_token_len = 96
args.output_token_len = 96
args.d_model = 1024
args.n_heads = 8
args.e_layers = 8
args.d_ff = 2048
args.dropout = 0.1
args.activation = 'relu'
args.use_norm = True
args.flash_attention = False
args.covariate = False
args.output_attention = False

# 初始化模型
model = timer_xl.Model(args)

# 加载模型权重
print("==> Loading checkpoint...")
model.load_state_dict(torch.load('checkpoints/checkpoint.pth'))
print("==> Checkpoint loaded.")

# 读取数据
print("==> Loading CSV data...")
df = pd.read_csv("./datasets/ETTh2.csv")
print("==> CSV loaded. Data shape:", df.shape)

# 准备输入数据
lookback_length = 1440  # 最长支持上下文长度为2880
input = torch.tensor(df["OT"][:lookback_length]).unsqueeze(0).float()
print("==> Input shape:", input.shape)

# 模型前向预测
prediction_length = 96
print("==> Running model...")
output = model(input.unsqueeze(-1), None, None)
print("==> Output shape:", output.shape)

# 可视化输入和模型输出
plt.figure(figsize=(12, 4))
plt.plot(input.squeeze().detach().numpy(), label="Input")
plt.plot(output.squeeze().detach().numpy(), label="Output")
plt.legend()
plt.grid()
plt.title("Input and Output Sequence")
plt.savefig("results/input_output.png")
plt.close()
print("==> Saved: results/input_output.png")

# 获取最终预测结果（只取最后96个）
pred = output[:, -96:, 0].squeeze().detach().numpy()

# 可视化真实数据与预测对比
plt.figure(figsize=(12, 4))
plt.plot(df["OT"][:lookback_length + prediction_length], color="limegreen", label="Groundtruth")
plt.plot(range(lookback_length, lookback_length + prediction_length), pred, color="tomato", label="Prediction")
plt.plot(df["OT"][:lookback_length], color="royalblue", label="Lookback")
plt.legend()
plt.grid()
plt.title("Groundtruth vs Prediction")
plt.savefig("results/prediction_result.png")
plt.close()
print("==> Saved: results/prediction_result.png")
