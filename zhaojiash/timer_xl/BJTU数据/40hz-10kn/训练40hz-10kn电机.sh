#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# =========================================================
# 1. 基础配置 (Basic Config)
# =========================================================
task_name="forecast"           # 训练时使用标准预测任务
model_id="BJTU40hz-10kn电机"  # 微调实验ID（区分原训练）
model_name="timer_xl"          # 模型架构（需与预训练权重匹配）

# 使用你的BJTU数据加载器
data_name="BJTU"

# =========================================================
# 2. 数据路径 (Data Paths)
# =========================================================
train_root_path="/home/ubuntu/zhaojia/OpenLTM_data_backup/datasets/BJTU_process_data/40Hz_-10kN_process_data/电机/train"
train_data_path=$(ls "$train_root_path" | grep "\.csv$" | head -n 1)

if [ -z "$train_data_path" ]; then
    echo "错误：在 $train_root_path 下未找到 .csv 文件！"
    exit 1
fi

# =========================================================
# 3. 模型参数 (Model Parameters) - 需与预训练模型保持一致
# =========================================================
n_vars=1             # 变量数（与你的数据匹配）
token_num=30         # 新增：与预训练保持一致的token数量
token_len=96         # Patch长度（需与预训练匹配）
seq_len=$[$token_num*$token_len]  # 计算序列长度（关键修正）
output_token_len=96  # 预测长度

# 以下参数根据预训练模型调整（参考scripts/adaptation/full_shot/timer_xl_etth1.sh）
d_model=1024         # 预训练模型的隐藏层维度
d_ff=2048            # 预训练模型的前馈网络维度
e_layers=8           # 预训练模型的层数
n_heads=8            # 注意力头数

# =========================================================
# 4. 微调超参数 (Hyperparameters)
# =========================================================
batch_size=256      # 批次大小（根据GPU内存调整）
train_epochs=10      # 微调轮数
learning_rate=5e-6   # 微调学习率

# =========================================================
# 5. 预训练权重路径 (Pretrained Model Path)
# =========================================================
pretrain_model_path="/home/ubuntu/zhaojia/checkpoints/Timer_xl/checkpoint.pth"  # 替换为实际路径

# =========================================================
# 6. 执行微调命令（关键修正：路径和参数格式）
# =========================================================
# 修正cd路径（确保OpenLTM的实际路径正确）
cd /home/ubuntu/zhaojia || {
    echo "错误：OpenLTM目录不存在，请检查路径！"
    exit 1
}

echo ">>>>>>> 开始微调: $model_id ($data_name)"

python run.py \
  --task_name "$task_name" \
  --is_training 1 \
  --model_id "$model_id" \
  --model "$model_name" \
  --data "$data_name" \
  --root_path "$train_root_path" \
  --data_path "$train_data_path" \
  --n_vars "$n_vars" \
  --seq_len "$seq_len" \
  --input_token_len "$token_len" \
  --output_token_len "$output_token_len" \
  --batch_size "$batch_size" \
  --train_epochs "$train_epochs" \
  --gpu 0 \
  --learning_rate "$learning_rate" \
  --d_model "$d_model" \
  --d_ff "$d_ff" \
  --e_layers "$e_layers" \
  --n_heads "$n_heads" \
  --use_norm \
  --adaptation \
  --pretrain_model_path "$pretrain_model_path"