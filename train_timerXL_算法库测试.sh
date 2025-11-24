#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# ==============================================================================
# --- 1. 训练配置区 (Training Configuration) ---
# ==============================================================================

# --- 任务与模型ID ---
task_name="forecast"
model_id="二维"  # [!!] 已更改 ID，以保存为新的检查点
model_name="timer_xl"          # 要使用的模型

# --- 训练数据路径 (Training Paths) ---
# [!!] 已更新为你的新路径
train_root_path="/home/ubuntu/hsz/OpenLTM_data_backup/datasets/算法库测试"
train_data_path="正常.csv"

# --- 核心模型参数 (Model Hyperparameters) ---
# [!!] 注意: 如果你的 "正常.csv" 只有1列，建议将 n_vars 改为 1
n_vars=2              
seq_len=32            
input_token_len=32    
output_token_len=8    

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
  --root_path "$train_root_path" \
  --data_path "$train_data_path" \
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