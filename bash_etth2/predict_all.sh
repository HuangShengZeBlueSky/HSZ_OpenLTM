python scripts/anomaly/compute_anomaly_from_residuals.py \
  --pred_np ./results/etth2_mv_96_24_pred.npy \
  --true_np ./results/etth2_mv_96_24_true.npy \
  --cols_json ./results/etth2_mv_96_24_cols.json \
  --method mad --alpha 3.5 \
  --out_csv ./results/etth2_mv_96_24_anomalies.csv