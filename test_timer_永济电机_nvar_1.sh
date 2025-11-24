#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# ==============================================================================
# --- 1. 测试配置区 (Testing Configuration) ---
# ==============================================================================

# --- 任务与模型ID ---
task_name="forecast_test_all" 
model_id="suanfaku_test_new"  # [!!] 确保这与你训练的 ID 一致
model_name="timer_xl"       # [!!] 确保这与你训练的 Model 一致

# --- 测试数据路径 (Testing Paths) ---
test_root_path="/home/ubuntu/hsz/OpenLTM_data_backup/datasets/永济电机轴承数据集/算法库测试/test"
test_data_path="" # 必须留空! 

# --- [!!] 新增：异常检测调参区 [!!] ---
# (你可以随意修改这里的 "mad", 3.0, 0.995, -1 来进行实验)

anom_method="mad"       # "mad" 或 "quantile"
anom_alpha=1.0          # MAD 阈值 (建议 2.5 ~ 4.0)。仅在 method="mad" 时有效。
anom_q=0.99          # 分位数阈值 (建议 0.99 ~ 0.999)。仅在 method="quantile" 时有效。
anom_only_idx=-1        # 检查哪个通道？ (-1 = 检查所有通道, 0 = 只检查第1个, 1 = 只检查第2个, ...)

# --- 核心模型参数 (Model Hyperparameters) ---
n_vars=1 
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
echo ">>>>>>> 1. 查找检查点 (Finding Checkpoint) for $model_id + $model_name ..."
CKPT_DIR=$(ls -1dt checkpoints/forecast_${model_id}_${model_name}_* | head -n1 | xargs -n1 basename)

if [ -z "$CKPT_DIR" ]; then
    echo "!!!!!! 错误：未找到与 "forecast_${model_id}_${model_name}" 匹配的检查点。测试中止。"
    exit 1
fi

echo ">>>>>>> 2. 开始一键测试 (Starting One-Key Test) using $CKPT_DIR"
echo ">>>>>>> 任务名称 (Task Name): $task_name"
echo ">>>>>>> 测试文件夹 (Testing folder): $test_root_path"
echo ">>>>>>> 异常检测参数 (Anomaly Params): method=$anom_method, alpha=$anom_alpha, q=$anom_q, only_idx=$anom_only_idx"
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
  --n_heads $n_heads \
  \
  # [!!] 新增：将异常检测参数传递给 run.py
  --anom_method $anom_method \
  --anom_alpha $anom_alpha \
  --anom_q $anom_q \
  --anom_only_idx $anom_only_idx

echo "========================================================"
echo ">>>>>>> 所有测试任务完成 (All testing finished.)"
echo "========================================================"