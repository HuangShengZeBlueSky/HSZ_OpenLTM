#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# ==============================================================================
# --- 1. 训练配置区 (Training Configuration) ---
# ==============================================================================

# --- 任务与模型ID ---
task_name="forecast"
model_id="suanfaku_test"  # [重要] 实验的唯一ID。测试时将通过此ID查找模型。
model_name="timer"      # 要使用的模型 (timer, timer_xl, moment, autotimes)

# --- 训练数据路径 (Training Paths) ---
# [!!] 已根据你的新路径更新
train_root_path="/home/ubuntu/hsz/OpenLTM_data_backup/datasets/永济电机轴承数据集/算法库测试/train"
train_data_path="train_data.csv"

# --- 核心模型参数 (Model Hyperparameters) ---
n_vars=2              # [!!] 必须与你的 .csv 文件中的列数（序列数）匹配
seq_len=32            # 训练时的输入长度
input_token_len=32    # Patch 长度
output_token_len=8    # 预测的 Patch 长度

# --- Transformer 架构参数 ---
d_model=512
d_ff=2048
e_layers=1
n_heads=8

# --- 训练与GPU参数 ---
batch_size=16
train_epochs=5
gpu_id=0
learning_rate=0.0001
weight_decay=0

# ==============================================================================
# --- 2. 训练执行区 (Training Execution) ---
# ==============================================================================
# [!!] 确保在正确的 OpenLTM 根目录 (run.py 所在的位置)
cd /home/ubuntu/hsz/OpenLTM 

echo ""
echo "========================================================"
echo ">>>>>>> 1. 开始训练 (Starting Training): $model_id"
echo ">>>>>>> 训练数据 (Training data): $train_root_path/$train_data_path"
echo "========================================================"

python run.py \
  --task_name $task_name \
  --is_training 1 \
  --model_id $model_id \
  --model $model_name \
  --data MultivariateDatasetBenchmark \
  --root_path $train_root_path \
  --data_path $train_data_path \
  --n_vars $n_vars \
  --seq_len $seq_len \
  --input_token_len $input_token_len \
  --output_token_len $output_token_len \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --gpu $gpu_id \
  --learning_rate $learning_rate \
  --weight_decay $weight_decay \
  --d_model $d_model \
  --d_ff $d_ff \
  --e_layers $e_layers \
  --n_heads $n_heads

if [ $? -eq 0 ]; then
    echo "========================================================"
    echo ">>>>>>> 训练完成 (Training Finished): $model_id"
    echo "========================================================"
else
    echo "!!!!!! 训练失败 (Training Failed) !!!!!!"
    exit 1
fi