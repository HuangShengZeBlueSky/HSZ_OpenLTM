#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# =========================================================
# 1. 基础配置 (必须与训练完全一致)
# =========================================================
task_name="forecast_test_all2" 
model_id="BJTU40hz_0kn齿轮箱"
model_name="timer_xl"
data_name="BJTU"

# =========================================================
# 2. 数据路径
# =========================================================
test_root_path="/home/ubuntu/zhaojia/OpenLTM_data_backup/datasets/BJTU_process_data/40Hz_0kN_process_data/齿轮箱/test"
test_data_path=""  # 留空，由数据加载器自动遍历

# =========================================================
# 3. 异常检测参数（可调）
# =========================================================
percentile=99.9  
vote_rate=0.01


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
# 4. 模型参数（⚠️ 必须与训练脚本完全一致！）
# =========================================================
n_vars=1
token_len=96
token_num=30
seq_len=$((token_num * token_len))   # = 2880？等等，你训练时写的是 30*96=2880，但下面又说4608？
# ⚠️ 注意：你训练脚本中 seq_len=$[$token_num*$token_len] → 30*96=2880
# 但你在测试脚本示例中写了 seq_len=4608 —— 这矛盾了！

# 因此这里应为：
input_token_len=96
output_token_len=96
test_seq_len=2880      # 通常与 seq_len 相同
test_pred_len=96

# ⚠️ 关键修正：模型结构参数必须匹配预训练/微调时的设置
d_model=1024    # ✅ 与训练一致
d_ff=2048       # ✅
e_layers=8      # ✅
n_heads=8       # ✅

# =========================================================
# 5. 运行参数
# =========================================================
batch_size=256
gpu_id=0

# =========================================================
# 6. 执行命令
# =========================================================
cd /home/ubuntu/zhaojia || {
    echo "错误：OpenLTM目录不存在，请检查路径！"
    exit 1
}

echo ">>>>>>> 1. 查找检查点..."
# 注意：检查点目录命名规则需与训练时 run.py 生成的一致
# 通常是：checkpoints/forecast_{model_id}_{model_name}_bs{batch}_lr{lr}_...
CKPT_DIR=$(ls -1dt checkpoints/forecast_${model_id}_${model_name}_* 2>/dev/null | head -n1)

if [ -z "$CKPT_DIR" ]; then
    echo "!!!!!! 错误：未找到检查点目录。"
    exit 1
fi

echo ">>>>>>> 使用检查点: $(basename "$CKPT_DIR")"

# 创建日志目录

LOG_DIR="all_tuning_logs"
mkdir -p "$LOG_DIR"  # 如果不存在就创建
echo ">>>>>>> 日志将保存到: $LOG_DIR/"

# 可选：总日志文件（取消注释下面一行即可启用）
TOTAL_LOG="$LOG_DIR/_${model_id}_${model_name}_all_experiments.log"

# =========================================================
# 6. 循环测试
# =========================================================
for param in "${PARAMS[@]}"; do
    IFS=',' read -r percentile vote_rate <<< "$param"

    echo "=========================================="
    echo ">>> 测试: model=${model_id}/${model_name}, p=${percentile}, v=${vote_rate}"
    echo "=========================================="
    
    # 写入分隔信息到总日志（通过 echo + tee）
    {
        echo ""
        echo "=================================================="
        echo ">>> 实验开始: p=${percentile}, v=${vote_rate}"
        echo ">>> Start time: $(date)"
        echo "=================================================="
    } | tee -a "$TOTAL_LOG"


    
    # 构建命令（注意：\ 后不能有空行！）
    python run.py \
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
      --percentile "$percentile" \
      --vote_rate "$vote_rate" \
      2>&1 | tee -a "$TOTAL_LOG"
    
    # 记录结束时间
    {
        echo ""
        echo ">>> End time: $(date)"
        echo ">>> 本次实验结束: p=${percentile}, v=${vote_rate}"
        echo ""
    } | tee -a "$TOTAL_LOG"

    echo ">>> 已记录到总日志: $TOTAL_LOG"
    echo ""
done

echo "✅ 所有调参实验完成！结果在 ./$LOG_DIR/ 目录下。"

