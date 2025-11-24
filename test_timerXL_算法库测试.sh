#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# ==============================================================================
# --- 1. 测试配置区 (Testing Configuration) ---
# ==============================================================================

# --- 任务与模型ID ---
# [!!] 关键: 使用 "forecast" 任务, 因为我们只测试单个文件
task_name="forecast_test_all" 
model_id="二维"  # [!!] 必须与你 train.sh 中使用的 model_id 完全一致！
model_name="timer_xl"          # 必须与 train.sh 中使用的 model_name 一致

# --- 测试数据路径 (Testing Paths) ---
# [!!] 已更新为你的新路径
test_root_path="/home/ubuntu/hsz/OpenLTM_data_backup/datasets/算法库测试"
test_data_path="1370_电腐蚀.csv" # [!!] 指定你想要的单个测试文件

# --- 核心模型参数 (Model Hyperparameters) ---
# [!!] 必须与你 train.sh 中使用的参数完全一致！
n_vars=2
seq_len=32
input_token_len=32
output_token_len=8
test_seq_len=32       
test_pred_len=8       

# --- Transformer 架构参数 ---
d_model=512
d_ff=2048
e_layers=1
n_heads=8

# --- GPU参数 ---
batch_size=16         
gpu_id=0

# ==============================================================================
# --- 2. 测试执行区 (Testing Execution) ---
# ==============================================================================
cd /home/ubuntu/hsz/OpenLTM 

echo ""
echo "========================================================"
echo ">>>>>>> 1. 查找检查点 (Finding Checkpoint) for $model_id..."
CKPT_DIR=$(ls -1dt checkpoints/*${model_id}* | head -n1 | xargs -n1 basename)

if [ -z "$CKPT_DIR" ]; then
    echo "!!!!!! 错误：未找到 $model_id 相关的检查点。测试中止。"
    exit 1
fi

echo ">>>>>>> 2. 开始测试 (Starting Test) using $CKPT_DIR"
echo ">>>>>>> 测试文件 (Test File): $test_root_path/$test_data_path"
echo "========================================================"

python run.py \
  --task_name $task_name \
  --is_training 0 \
  --model_id $model_id \
  --model $model_name \
  --data MultivariateDatasetBenchmark \
  --root_path "$test_root_path" \
  --data_path "$test_data_path" \
  --n_vars $n_vars \
  --seq_len $seq_len \
  --input_token_len $input_token_len \
  --output_token_len $output_token_len \
  --test_seq_len $test_seq_len \
  --test_pred_len $test_pred_len \
  --batch_size $batch_size \
  --gpu $gpu_id \
  --test_dir "$CKPT_DIR" \
  --test_file_name "checkpoint.pth" \
  --d_model $d_model \
  --d_ff $d_ff \
  --e_layers $e_layers \
  --n_heads $n_heads

echo "========================================================"
echo ">>>>>>> 测试完成 (Testing finished.)"
echo "========================================================"