#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# =========================================================
# 1. 基础配置
# =========================================================
model_id="工况1_转速12转每秒_微调"
model_name="moirai"
setting="moirai_anomaly_test"  # 对应 test_moirai_anomaly.py 中的 --setting

# =========================================================
# 2. 数据路径
# =========================================================
test_root_path="/home/ubuntu/zhaojia/OpenLTM_data_backup/datasets/中车研究院实验台数据集/工况1_转速12转每秒/test"

# =========================================================
# 3. 检查点路径（自动查找最新）
# =========================================================
cd /home/ubuntu/zhaojia || {
    echo "错误：工作目录不存在！"
    exit 1
}

CKPT_DIR=$(ls -1dt checkpoints/moirai-finetuned* 2>/dev/null | head -n1)
if [ -z "$CKPT_DIR" ]; then
    echo "!!!!!! 错误：未找到 Moirai 微调检查点目录（应以 moirai-finetuned开头）"
    exit 1
fi
echo ">>>>>>> 使用检查点: $(basename "$CKPT_DIR")"

# =========================================================
# 4. 模型与数据参数（⚠️ 必须与训练时一致！）
# =========================================================
context_len=496
pred_len=16
patch_size=16
batch_size=64
gpu_id=0

# =========================================================
# 5. 调参组合
# =========================================================
declare -a PARAMS=(
    "99.9,0.001"
    "99.9,0.005"
    "99.9,0.01"
    "99.9,0.015"
    "99.9,0.05"
    "99.5,0.001"
    "99.5,0.005"
    "99.5,0.01"
    "99.5,0.015"
    "99.5,0.05"
    "99.0,0.001"
    "99.0,0.005"
    "99.0,0.01"
    "99.0,0.015"
    "99.0,0.05"
    "98.5,0.001"
    "98.5,0.005"
    "98.5,0.01"
    "98.5,0.015"
    "98.5,0.05"
)

# =========================================================
# 6. 日志设置
# =========================================================
LOG_DIR="all_tuning_logs"
mkdir -p "$LOG_DIR"
TOTAL_LOG="$LOG_DIR/_${model_id}_${model_name}_all_experiments.log"

# =========================================================
# 7. 循环执行
# =========================================================
for param in "${PARAMS[@]}"; do
    IFS=',' read -r percentile vote_rate <<< "$param"

    echo "=========================================="
    echo ">>> 测试: p=${percentile}, v=${vote_rate}"
    echo "=========================================="

    # 记录实验开始
    {
        echo ""
        echo "=================================================="
        echo ">>> 实验开始: p=${percentile}, v=${vote_rate}"
        echo ">>> Checkpoint: $(basename "$CKPT_DIR")"
        echo ">>> Start time: $(date)"
        echo "=================================================="
    } | tee -a "$TOTAL_LOG"

    # 构建 setting 名称（避免非法字符）
    safe_setting="${model_id}_${model_name}_p${percentile}_v${vote_rate}"
    safe_setting=$(echo "$safe_setting" | sed 's/[^\w\-\.]/_/g')

    # 执行测试脚本
    python run_moirai.py \
      --checkpoint_dir "$CKPT_DIR" \
      --test_root "$test_root_path" \
      --setting "$safe_setting" \
      --context_len "$context_len" \
      --pred_len "$pred_len" \
      --patch_size "$patch_size" \
      --batch_size "$batch_size" \
      --gpu "$gpu_id" \
      --percentile "$percentile" \
      --vote_rate "$vote_rate" \
      2>&1 | tee -a "$TOTAL_LOG"

    # 记录结束
    {
        echo ""
        echo ">>> End time: $(date)"
        echo ">>> 本次实验结束: p=${percentile}, v=${vote_rate}"
        echo ""
    } | tee -a "$TOTAL_LOG"

    echo ">>> 已记录到总日志: $TOTAL_LOG"
    echo ""
done

echo "✅ 所有 Moirai 异常检测调参实验完成！结果在 ./$LOG_DIR/ 目录下。"