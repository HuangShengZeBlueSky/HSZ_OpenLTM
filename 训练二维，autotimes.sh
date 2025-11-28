#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# =========================================================
# 1. 基础配置 (Basic Config)
# =========================================================
task_name="forecast"           # 标准预测任务用于训练
model_id="工1转12防止过拟合"   # [修改] 新名字，避免覆盖
model_name="autotimes"          # 架构名保持不变，我们通过参数控制大小
data_name="BJTU"               # 使用我们修复好的 BJTU 加载器
llm_model="OPT"
# =========================================================
# 2. 数据路径 (Data Paths)
# =========================================================
train_root_path="/home/ubuntu/hsz/OpenLTM_data_backup/datasets/中车研究院实验台数据集/工况1_转速12转每秒/train"

# 自动查找该文件夹下的 .csv 文件名
train_data_path=$(ls "$train_root_path" | grep "\.csv$" | head -n 1)

if [ -z "$train_data_path" ]; then
    echo "错误：在 $train_root_path 下未找到 .csv 文件！"
    exit 1
fi

# =========================================================
# 3. 模型参数 (Small Version) - [核心修改]
# =========================================================
n_vars=2              # 变量数
seq_len=4608          # 输入长度
input_token_len=96    # Patch 长度
output_token_len=96   # 预测长度

# [!!] 瘦身计划：大幅减小模型容量，防止过拟合异常 [!!]
d_model=64            # 原来 256 -> 改为 64
d_ff=128              # 原来 512 -> 改为 128
e_layers=2            # 原来 4   -> 改为 2
n_heads=4             # 原来 8   -> 改为 4

# =========================================================
# 4. 训练超参数
# =========================================================
batch_size=32       
train_epochs=15        
learning_rate=0.001   

# =========================================================
# 5. 执行命令
# =========================================================
cd /home/ubuntu/hsz/OpenLTM 

echo ">>>>>>> 开始训练 (Small版): $model_id"

python run.py \
  --task_name $task_name \
  --is_training 1 \
  --model_id $model_id \
  --model $model_name \
  --data $data_name \
  --root_path "$train_root_path" \
  --data_path "$train_data_path" \
  --n_vars $n_vars \
  --seq_len $seq_len \
  --input_token_len $input_token_len \
  --output_token_len $output_token_len \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --gpu 0 \
  --learning_rate $learning_rate \
  --d_model $d_model \
  --d_ff $d_ff \
  --e_layers $e_layers \
  --n_heads $n_heads\
  --llm_model $llm_model