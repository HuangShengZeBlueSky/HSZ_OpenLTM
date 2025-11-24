#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# =========================================================
# 1. 基础配置 (需与训练一致)
# =========================================================
# [!!] 关键：使用自定义的“一键测试”任务
task_name="forecast_test_all2" 

model_id="工况1—转速1370转每分钟"     # 必须与训练一致
model_name="timer_xl"          # 必须与训练一致
data_name="BJTU"               # 必须与训练一致

# =========================================================
# 2. 数据路径
# =========================================================
# 指向包含所有测试 .csv 的文件夹
test_root_path="/home/ubuntu/zhaojia/永济电机轴承数据集/工况1_转速为1370转每分钟/test"
test_data_path="" # 留空，脚本会自动遍历文件夹

# =========================================================
# 3. 异常检测参数 - [核心可调参数]
# =========================================================
# 决定异常判定的灵敏度
# 99.9 = 非常严格 (误报极低，只报大故障) -> 结果类似 30%
# 99.0 = 严格
# 95.0 = 敏感 (误报增加，能抓小故障) -> 结果可能飙升到 80%
percentile=99.9  

# =========================================================
# 4. 模型参数 (必须与训练完全一致，不可改)
# =========================================================
n_vars=3
seq_len=4608
input_token_len=96
output_token_len=96
test_seq_len=4608
test_pred_len=96
d_model=256
d_ff=512
e_layers=2
n_heads=8

# =========================================================
# 5. 运行参数
# =========================================================
batch_size=2048  # 测试时可以用更大的 batch，只要显存够
gpu_id=0

# =========================================================
# 6. 执行命令
# =========================================================
cd /home/ubuntu/zhaojia/OpenLTM 

echo ">>>>>>> 1. 查找检查点..."
CKPT_DIR=$(ls -1dt checkpoints/forecast_${model_id}_${model_name}_* | head -n1 | xargs -n1 basename)

if [ -z "$CKPT_DIR" ]; then
    echo "!!!!!! 错误：未找到检查点。"
    exit 1
fi

echo ">>>>>>> 2. 开始 BJTU 复刻版测试"
echo ">>>>>>> 策略: 验证集校准 + 大步长跳跃 + ${percentile}% 阈值"

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