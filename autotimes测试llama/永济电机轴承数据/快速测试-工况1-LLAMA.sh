#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

# =========================================================
# 1. 基础配置
# =========================================================
# ⚠️ 关键修改：指向新写的支持“一次推理，内部寻优”的类
task_name="forecast_test_all3" 

model_id="永济工况1_转速1370转每秒LLAMA"
model_name="autotimes"
data_name="BJTU"
llm_model="LLAMA"
local_path="/raid/hsz/models"

# =========================================================
# 2. 数据路径
# =========================================================
test_root_path="/raid/hsz/OpenLTM_data_backup/datasets/永济电机轴承数据集/工况1_转速为1370转每分钟/test/"
# test_data_path 留空

# =========================================================
# 3. 模型参数 (保持原脚本配置)
# =========================================================
n_vars=3           # ✅ 保持为 3
input_token_len=96
output_token_len=96
seq_len=2880   
test_seq_len=2880
test_pred_len=96
d_model=1024
d_ff=2048
e_layers=8
n_heads=8

# =========================================================
# 4. 运行参数
# =========================================================
batch_size=16
gpu_id=0

# =========================================================
# 5. 执行命令 (自动寻优版)
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
# 日志文件名带上 GridSearch 标识
TOTAL_LOG="$LOG_DIR/GridSearch_${model_id}_${model_name}.log"

{
    echo "=================================================="
    echo ">>> 极速寻优实验开始 (Internal Grid Search)"
    echo ">>> Target: 永济工况1_转速1370"
    echo ">>> Start time: $(date)"
    echo "=================================================="
} | tee -a "$TOTAL_LOG"

# >>> 核心修改 <<<
# 1. 加上 -u 防止日志阻塞
# 2. 去掉了循环
# 3. 去掉了 --percentile 和 --vote_rate (Python 内部接管)
# 4. 确保 local_path 后面加了反斜杠
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

{
    echo ""
    echo ">>> End time: $(date)"
    echo ">>> 结果已保存至: $TOTAL_LOG"
} | tee -a "$TOTAL_LOG"

echo "✅ 完成。请打开日志文件查看最佳参数排行榜。"