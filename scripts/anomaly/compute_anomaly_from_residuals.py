import argparse
import os
import json
import numpy as np
import pandas as pd

EPS = 1e-8

def mad_flags(residuals: np.ndarray, alpha: float = 3.5) -> np.ndarray:
    """
    residuals: (M, C), 非负（建议为 |y - y_hat|）
    基于稳健 Z 分数（median + 1.4826*MAD）逐列阈值
    返回布尔矩阵 (M, C)
    """
    med = np.median(residuals, axis=0, keepdims=True)
    mad = np.median(np.abs(residuals - med), axis=0, keepdims=True) + EPS
    robust_z = np.abs(residuals - med) / (1.4826 * mad)
    return robust_z > alpha

def quantile_flags(residuals: np.ndarray, q: float = 0.995) -> np.ndarray:
    """
    residuals: (M, C)
    基于每列分位数阈值
    """
    thr = np.quantile(residuals, q, axis=0, keepdims=True)
    return residuals > thr

def main():
    ap = argparse.ArgumentParser(description="Compute residual-based anomalies and print anomaly rate.")
    ap.add_argument("--pred_np", required=True, help="预测 .npy, 形状 (N,T,C) 或 (N,C)")
    ap.add_argument("--true_np", required=True, help="真值 .npy, 形状 (N,T,C) 或 (N,C)")
    ap.add_argument("--cols_json", default="", help="列名 JSON（如 ['HUFL',..., 'OT']）")
    ap.add_argument("--method", choices=["mad", "quantile"], default="mad")
    ap.add_argument("--alpha", type=float, default=3.5, help="MAD 乘子（mad 方法）")
    ap.add_argument("--q", type=float, default=0.995, help="分位数（quantile 方法）")
    ap.add_argument("--only_idx", type=int, default=-1, help="仅检测某列索引（0 开始，-1 表示全部列）")
    ap.add_argument("--out_csv", default="./results/anomalies.csv")
    args = ap.parse_args()

    preds = np.load(args.pred_np, allow_pickle=False)
    trues = np.load(args.true_np, allow_pickle=False)

    orig_ndim = preds.ndim
    if orig_ndim == 3:
        N, T, C = preds.shape
        preds_f = preds.reshape(N * T, C)
        trues_f = trues.reshape(N * T, C)
    elif orig_ndim == 2:
        # 无法计算 sample_idx/horizon，只能逐行
        C = preds.shape[1]
        N, T = None, None
        preds_f = preds
        trues_f = trues
    else:
        raise ValueError("Unsupported pred/true shape; expect (N,T,C) or (N,C)")

    # 计算残差（L1）
    residuals = np.abs(trues_f - preds_f).astype(np.float32)  # (M, C)
    M = residuals.shape[0]

    # 列名
    if args.cols_json and os.path.isfile(args.cols_json):
        with open(args.cols_json, "r") as f:
            all_cols = json.load(f)
        if not isinstance(all_cols, list):
            raise ValueError("cols_json 必须为列表 JSON")
        if len(all_cols) != C:
            # 保护：列名数量与 C 不一致时，回退自动列名
            all_cols = [f"var_{i}" for i in range(C)]
    else:
        all_cols = [f"var_{i}" for i in range(C)]

    # 仅检测单列
    if args.only_idx >= 0:
        if args.only_idx < 0 or args.only_idx >= C:
            raise ValueError(f"only_idx 越界：{args.only_idx} 不在 [0,{C-1}]")
        residuals_used = residuals[:, [args.only_idx]]
        used_cols = [all_cols[args.only_idx]]
    else:
        residuals_used = residuals
        used_cols = list(all_cols)

    # 阈值判定
    if args.method == "mad":
        flags = mad_flags(residuals_used, alpha=args.alpha)
    else:
        flags = quantile_flags(residuals_used, q=args.q)
    # flags: (M, C_used) 布尔

    # 组织输出
    if orig_ndim == 3 and N is not None and T is not None:
        sample_idx = np.arange(M) // T
        horizon = np.arange(M) % T
    else:
        sample_idx = np.arange(M)
        horizon = np.full(M, -1, dtype=int)

    df = pd.DataFrame({"sample_idx": sample_idx, "horizon": horizon})
    for j, c in enumerate(used_cols):
        df[f"anom_{c}"] = flags[:, j].astype(np.int32)
        df[f"resid_{c}"] = residuals_used[:, j]

    # 保存 CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # 计算并输出异常率（样本级：任一 horizon/变量 异常即计为异常样本）
    # 行级（任一列异常）
    row_any = flags.any(axis=1) if flags.ndim == 2 else flags.astype(bool)
    if orig_ndim == 3 and N is not None and T is not None:
        # 聚合到样本级
        sample_any = pd.Series(row_any).groupby(sample_idx).any()
        sample_rate = float(sample_any.mean())
        print(f"{sample_rate:.6f}")
    else:
        # 无法聚合样本，输出行级异常率
        row_rate = float(row_any.mean())
        print(f"{row_rate:.6f}")

if __name__ == "__main__":
    main()