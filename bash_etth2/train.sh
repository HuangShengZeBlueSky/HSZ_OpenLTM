cd /home/ubuntu/hsz/OpenLTM

python run.py \
  --task_name forecast \
  --is_training 1 \
  --model_id etth2_mv_96_24 \
  --model timer_xl \
  --data ETTh2 \
  --root_path ./datasets \
  --data_path ETTh2.csv \
  --seq_len 96 \
  --input_token_len 96 \
  --output_token_len 24 \
  --test_seq_len 96 \
  --test_pred_len 24 \
  --n_vars 7 \
  --batch_size 32 \
  --train_epochs 10 \
  --gpu 0