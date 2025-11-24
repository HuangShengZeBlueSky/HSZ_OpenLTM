#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# =========================================================
# 1. 基础配置
# =========================================================
task_name="forecast_test_all2" # 使用我们修改后的点级检测类

model_id="工1转12防止过拟合"   # [修改] 对应训练时的名字
model_name="timer_xl"          
data_name="BJTU"               

# =========================================================
# 2. 数据路径
# =========================================================
test_root_path="/home/ubuntu/hsz/OpenLTM_data_backup/datasets/中车研究院实验台数据集/工况1_转速12转每秒/test"
test_data_path="" # 自动遍历

# =========================================================
# 3. 异常检测参数
# =========================================================
# 依然先尝试 99.9，看看小模型是否拉大了正常和异常的差距
percentile=99  

# =========================================================
# 4. 模型参数 (必须与训练脚本完全一致!)
# =========================================================
n_vars=2
seq_len=4608
input_token_len=96
output_token_len=96
test_seq_len=4608
test_pred_len=96

# [!!] 必须一致 [!!]
d_model=64
d_ff=128
e_layers=2
n_heads=4

# =========================================================
# 5. 运行参数
# =========================================================
batch_size=2048 
gpu_id=0

# =========================================================
# 6. 执行命令
# =========================================================
cd /home/ubuntu/hsz/OpenLTM 

echo ">>>>>>> 1. 查找检查点 (Small版)..."
# 自动找包含 Small 的检查点
CKPT_DIR=$(ls -1dt checkpoints/forecast_${model_id}_${model_name}_* | head -n1 | xargs -n1 basename)

if [ -z "$CKPT_DIR" ]; then
    echo "!!!!!! 错误：未找到检查点。请先运行训练脚本！"
    exit 1
fi

echo ">>>>>>> 找到模型: $CKPT_DIR"
echo ">>>>>>> 2. 开始测试 (Small Model)"

python run.py \
  --task_name $task_name \
  --is_training 0 \
  --model_id $model_id \
  --model $model_name \
  --data $data_name \
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
  --n_heads $n_heads \
  --percentile $percentile