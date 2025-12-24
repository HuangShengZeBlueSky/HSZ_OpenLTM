#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

# =========================================================
# 1. 基础配置
# =========================================================
# ⚠️ 注意：请确保你在 run.py 或 exp/__init__.py 中
# 将这个 task_name 映射到了你的新类 Exp_Forecast_TestAll3
task_name="forecast_test_all3" 

model_id="工况1_转速12转每秒_微调"
model_name="autotimes"
data_name="BJTU"
llm_model="OPT"
local_path="/raid/hsz/models"

# =========================================================
# 2. 数据路径
# =========================================================
test_root_path="/raid/hsz/OpenLTM_data_backup/datasets/中车研究院实验台数据集/工况1_转速12转每秒/test/"
# test_data_path 留空，由数据加载器自动遍历

# =========================================================
# 3. 模型参数 (保持与训练一致)
# =========================================================
n_vars=2
input_token_len=96
output_token_len=96
seq_len=2880       # 训练时的 seq_len
test_seq_len=2880  # 测试时通常保持一致
test_pred_len=96

d_model=1024
d_ff=2048
e_layers=8
n_heads=8

# =========================================================
# 4. 运行配置
# =========================================================
batch_size=1024
gpu_id=0

# =========================================================
# 5. 准备工作
# =========================================================
echo ">>>>>>> 1. 查找检查点..."
CKPT_DIR=$(ls -1dt checkpoints/forecast_${model_id}_${model_name}_* 2>/dev/null | head -n1)

if [ -z "$CKPT_DIR" ]; then
    echo "!!!!!! 错误：未找到检查点目录。"
    exit 1
fi

echo ">>>>>>> 使用检查点: $(basename "$CKPT_DIR")"

# 创建日志目录
LOG_DIR="all_tuning_logs"
mkdir -p "$LOG_DIR"
TOTAL_LOG="$LOG_DIR/GridSearch_${model_id}_${model_name}.log"

echo ">>>>>>> 日志将保存到: $TOTAL_LOG"

# =========================================================
# 6. 执行单次运行 (内部包含网格搜索)
# =========================================================

# 记录开始时间
{
    echo "=================================================="
    echo ">>> 极速寻优实验开始 (Internal Grid Search)"
    echo ">>> Start time: $(date)"
    echo "=================================================="
} | tee -a "$TOTAL_LOG"

# >>> 核心命令 <<<
# 1. 使用 python -u 防止日志缓存
# 2. 移除了 --percentile 和 --vote_rate (由 Python 内部接管)
# 3. 只需要跑一次
python -u run.py \
  --task_name "$task_name" \
  --is_training 0 \
  --model_id "$model_id" \
  --model "$model_name" \
  --data "$data_name" \
  --root_path "$test_root_path" \
  --data_path "" \
  --n_vars "$n_vars" \
  --seq_len "$seq_len" \
  --input_token_len "$input_token_len" \
  --output_token_len "$output_token_len" \
  --test_seq_len "$test_seq_len" \
  --test_pred_len "$test_pred_len" \
  --batch_size "$batch_size" \
  --gpu "$gpu_id" \
  --test_dir "$(basename "$CKPT_DIR")" \
  --test_file_name "checkpoint.pth" \
  --d_model "$d_model" \
  --d_ff "$d_ff" \
  --e_layers "$e_layers" \
  --n_heads "$n_heads" \
  --llm_model "$llm_model" \
  --local_path "$local_path" \
  2>&1 | tee -a "$TOTAL_LOG"

# 记录结束时间
{
    echo ""
    echo ">>> End time: $(date)"
    echo ">>> 实验结束，请查看上方表格获取最佳参数。"
    echo "=================================================="
} | tee -a "$TOTAL_LOG"

echo "✅ 完成。请打开日志文件查看最佳参数排行榜。"