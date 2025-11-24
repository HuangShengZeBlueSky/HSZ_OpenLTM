#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# =========================================================
# 1. 基础配置 (Basic Config)
# =========================================================
task_name="forecast"           # 训练时使用标准预测任务
model_id="工况2转速14中等模型"     # 实验ID (生成的文件夹名包含它)
model_name="timer_xl"          # 模型架构

# [!!] 关键：指定使用我们新写的 BJTU 加载器
data_name="BJTU"

# =========================================================
# 2. 数据路径 (Data Paths)
# =========================================================
# 训练集所在的文件夹
train_root_path="/home/ubuntu/hsz/OpenLTM_data_backup/datasets/中车研究院实验台数据集/工况2_转速14转每秒/train"

# 自动查找该文件夹下的 .csv 文件名
train_data_path=$(ls "$train_root_path" | grep "\.csv$" | head -n 1)

if [ -z "$train_data_path" ]; then
    echo "错误：在 $train_root_path 下未找到 .csv 文件！"
    exit 1
fi

# =========================================================
# 3. 模型参数 (Model Parameters) - [训练核心]
# =========================================================
n_vars=2              # 变量数 (你的数据是2列)
seq_len=4608          # 输入序列长度 (长窗口)
input_token_len=96    # Patch 长度 (切片大小)
output_token_len=96   # 预测长度 (Timer需预测未来)

d_model=256           # 隐藏层维度
d_ff=512              # 前馈网络维度
e_layers=2            # 层数
n_heads=8             # 注意力头数

# =========================================================
# 4. 训练超参数 (Hyperparameters) - [可调]
# =========================================================
batch_size=2048         # 批次大小
train_epochs=5        # 训练轮数
learning_rate=0.001   # 学习率

# =========================================================
# 5. 执行命令
# =========================================================
cd /home/ubuntu/hsz/OpenLTM 

echo ">>>>>>> 开始训练: $model_id ($data_name)"

python run.py \
  --task_name $task_name \
  --is_training 1 \
  --model_id $model_id \
  --model $model_name \
  --data $data_name \
  --root_path "$train_root_path" \
  --data_path "$train_data_path" \
  --n_vars $n_vars \
  --seq_len $seq_len \
  --input_token_len $input_token_len \
  --output_token_len $output_token_len \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --gpu 0 \
  --learning_rate $learning_rate \
  --d_model $d_model \
  --d_ff $d_ff \
  --e_layers $e_layers \
  --n_heads $n_heads