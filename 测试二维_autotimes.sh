#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# =========================================================
# 1. 基础配置 (需与训练脚本完全一致)
# =========================================================
task_name="forecast_test_all2"
model_id="工1转12防止过拟合"   # 必须与训练时一致
model_name="autotimes"
data_name="BJTU"
llm_model="OPT"

# =========================================================
# 2. 测试数据路径 (这里换成你想测试的异常数据)
# =========================================================
# 例如：测试电腐蚀数据
test_root_path="/home/ubuntu/hsz/OpenLTM_data_backup/datasets/中车研究院实验台数据集/工况1_转速12转每秒/test"
# 自动找一个 csv，或者你可以手动指定具体文件名
test_data_path=$(ls "$test_root_path" | grep "\.csv$" | head -n 1)
# test_data_path="1370_电腐蚀.csv"  # 如果你想测特定文件，取消注释这行

if [ -z "$test_data_path" ]; then
    echo "错误：在 $test_root_path 下未找到 .csv 文件！"
    exit 1
fi

# =========================================================
# 3. 模型参数 (必须与训练脚本完全一致)
# =========================================================
n_vars=2
seq_len=4608
input_token_len=96
output_token_len=96

# [!!] 必须与训练时的瘦身参数一致 [!!]
d_model=64
d_ff=128
e_layers=2
n_heads=4

# =========================================================
# 4. 自动查找 Checkpoint
# =========================================================
# 构造训练时生成的目录名模式
# 注意：这里匹配的是 run.py 生成的 setting 字符串
# 格式: task_model_id_model_data_sl_it_ot_lr_bt_wd_el_dm_dff_nh_cos_test
# 我们用通配符模糊匹配，找到最新的那个
checkpoints_dir="/home/ubuntu/hsz/OpenLTM/checkpoints"
pattern="forecast_${model_id}_${model_name}_${data_name}_sl${seq_len}_it${input_token_len}_ot${output_token_len}_lr*_bt*_wd*_el${e_layers}_dm${d_model}_dff${d_ff}_nh${n_heads}_*"

# 找到最新的目录
test_dir=$(ls -td ${checkpoints_dir}/${pattern} | head -n 1 | xargs basename)

if [ -z "$test_dir" ]; then
    echo "错误：未找到匹配的 Checkpoint 目录！请检查 pattern 或确认训练是否成功保存。"
    echo "搜索模式: $pattern"
    exit 1
fi

echo ">>>>>>> 使用 Checkpoint: $test_dir"

# =========================================================
# 5. 执行测试
# =========================================================
cd /home/ubuntu/hsz/OpenLTM

# 异常检测参数 (可选)
# --anom_method mad --anom_alpha 4.0
percentile=99  
vote_rate=0.01

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
  --batch_size 32 \
  --gpu 0 \
  --d_model $d_model \
  --d_ff $d_ff \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --llm_model $llm_model \
  --test_dir "$test_dir" \
  --test_file_name checkpoint.pth