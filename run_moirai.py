# test_moirai_anomaly.py
import os
import argparse
import torch
import numpy as np
import json
from safetensors.torch import load_file
import joblib
from torch.utils.data import DataLoader

# --- 自定义模块 ---
from data_provider.data_loader import BJTUAnomalyloader
from uni2ts.model.moirai import MoiraiModule
from uni2ts.distribution import StudentTOutput


# ===============================
#           数据集包装（与训练一致）
# ===============================
class MoiraiFinetuneDataset:
    def __init__(self, bjtu_loader, context_len, pred_len, patch_size):
        self.bjtu_loader = bjtu_loader
        self.context_len = context_len
        self.pred_len = pred_len
        self.total_len = context_len + pred_len
        self.patch_size = patch_size

        sample_x, _, _, _ = bjtu_loader[0]
        self.variate_dim = sample_x.shape[1]

    def __len__(self):
        return len(self.bjtu_loader)

    def __getitem__(self, idx):
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.bjtu_loader[idx]
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        L_orig, C_orig = seq_x.shape

        L_target = self.total_len
        C_target = self.variate_dim

        if L_orig < L_target:
            pad = torch.zeros(L_target - L_orig, C_orig)
            seq_x = torch.cat([seq_x, pad], dim=0)
        elif L_orig > L_target:
            seq_x = seq_x[:L_target]

        if C_orig < C_target:
            pad = torch.zeros(L_target, C_target - C_orig)
            seq_x = torch.cat([seq_x, pad], dim=1)
        elif C_orig > C_target:
            seq_x = seq_x[:, :C_target]

        L, C = L_target, C_target
        target = seq_x.unsqueeze(-1)  # (L, C, 1)
        observed_mask = torch.ones(L, C, 1, dtype=torch.bool)
        prediction_mask = torch.zeros(L, C, dtype=torch.bool)
        prediction_mask[self.context_len:, :] = True

        sample_id = torch.full((L, C), idx, dtype=torch.long)
        time_id = torch.arange(L).unsqueeze(1).repeat(1, C)
        variate_id = torch.arange(C).unsqueeze(0).repeat(L, 1)
        patch_id = (time_id // self.patch_size).long()

        return {
            "target": target,
            "observed_mask": observed_mask,
            "prediction_mask": prediction_mask,
            "sample_id": sample_id,
            "time_id": time_id,
            "variate_id": variate_id,
            "patch_id": patch_id,
            "patch_size": torch.tensor(self.patch_size),
        }


def collate_fn(batch):
    out = {}
    for k in batch[0]:
        if k == "patch_size":
            out[k] = batch[0][k]
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


# ===============================
#           模型加载
# ===============================
def load_finetuned_moirai(checkpoint_dir, device):
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    raw_patch_size = config.get('patch_size', 16)
    model = MoiraiModule(
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        patch_sizes=[raw_patch_size],
        max_seq_len=config['max_seq_len'],
        distr_output=StudentTOutput(),
        attn_dropout_p=config.get('attn_dropout_p', 0.0),
        dropout_p=config.get('dropout_p', 0.0),
        scaling=config.get('scaling', True),
    )

    # 尝试加载 safetensors
    model_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(model_path):
        state_dict = load_file(model_path, device="cpu")
    else:
        # 尝试加载 .pth
        model_path = os.path.join(checkpoint_dir, "final_model.pth")
        state_dict = torch.load(model_path, map_location="cpu")

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"✅ Loaded Moirai model from {checkpoint_dir}")
    return model


# ===============================
#           NLL 计算
# ===============================
def compute_nll_loss(model, batch):
    with torch.no_grad():
        # Forward pass returns a distribution aligned with original time steps
        distr = model(
            target=batch["target"],
            observed_mask=batch["observed_mask"],
            sample_id=batch["sample_id"],
            time_id=batch["time_id"],
            variate_id=batch["variate_id"],
            prediction_mask=batch["prediction_mask"],
            patch_size=batch["patch_size"]
        )

        # Use the original target WITHOUT squeezing or reshaping
        target = batch["target"]  # Shape: (B, L, C, 1)

        # Some versions of Uni2TS expect (B, L, C), others (B, L, C, 1)
        # Try both: if log_prob fails with (B,L,C,1), squeeze last dim

        try:
            nll = -distr.log_prob(target)
        except ValueError as e:
            if "broadcastable" in str(e):
                # Try squeezing the last dimension
                target_squeezed = target.squeeze(-1)  # (B, L, C)
                nll = -distr.log_prob(target_squeezed)
            else:
                raise e

        # Apply mask: observed_mask is (B, L, C, 1)
        observed_mask = batch["observed_mask"].float()
        nll = nll * observed_mask.squeeze(-1) if nll.dim() == 3 else nll * observed_mask

        return nll


# ===============================
#           主函数
# ===============================
def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # --- 加载模型 ---
    model = load_finetuned_moirai(args.checkpoint_dir, device)

    # --- 加载 scaler（用于测试集标准化）---
    scaler_path = os.path.join(args.checkpoint_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("✅ Loaded scaler")
    else:
        scaler = None
        print("⚠️ No scaler found!")

    # --- 设置路径 ---
    base_data_dir = os.path.dirname(args.test_root.rstrip('/'))  # 去掉 /test
    val_data_path = os.path.join(base_data_dir, "train", "正常运行0-1.1_170623135209.csv")
    test_dir = args.test_root  # e.g., ".../test/"

    # --- 构建验证集 ---
    print("🔍 Building validation dataset...")
    val_loader_bjtu = BJTUAnomalyloader(
        root_path=val_data_path,
        seq_len=args.context_len + args.pred_len,
        patch_len=args.patch_size,
        flag="valid",
        scaler=scaler
    )
    val_dataset = MoiraiFinetuneDataset(
        val_loader_bjtu, args.context_len, args.pred_len, args.patch_size
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2
    )

    # --- 校准阶段：计算 NLL 阈值 ---
    print("📊 Calibrating NLL threshold on validation set...")
    all_val_nlls = []
    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        nll = compute_nll_loss(model, batch)  # (B, L, C)
        pred_nll = nll[:, args.context_len:, :]  # 只取预测部分
        point_nll = pred_nll.mean(dim=-1)        # (B, PRED)
        all_val_nlls.append(point_nll.cpu().numpy().flatten())

    all_val_nlls = np.concatenate(all_val_nlls)
    thr_nll = np.percentile(all_val_nlls, args.percentile)
    print(f"🎯 NLL threshold ({args.percentile}%): {thr_nll:.6f}")

    # --- 测试阶段 ---
    # out_root = os.path.join("/test_results", args.setting)
    # os.makedirs(out_root, exist_ok=True)


    # --- 测试阶段 ---
    import re
    # 安全处理 setting 名称，防止路径穿越或非法字符
    safe_setting = re.sub(r'[^\w\-\._]', '_', args.setting.strip('/'))
    if not safe_setting:
        safe_setting = "default_run"
    out_root = os.path.join("test_results", safe_setting)
    os.makedirs(out_root, exist_ok=True)
    print(f"📁 Saving results to: {os.path.abspath(out_root)}")

    result_file = open(os.path.join(out_root, "result_summary.txt"), 'a')

    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".csv")])
    print(f"🧪 Found {len(test_files)} test files in {test_dir}")

    for csv_file in test_files:
        print(f"\n>>> Testing: {csv_file}")
        full_path = os.path.join(test_dir, csv_file)

        test_loader_bjtu = BJTUAnomalyloader(
            root_path=full_path,
            test_data_path=full_path, 
            seq_len=args.context_len + args.pred_len,
            patch_len=args.patch_size,
            flag="test",
            scaler=scaler
        )
        test_dataset = MoiraiFinetuneDataset(
            test_loader_bjtu, args.context_len, args.pred_len, args.patch_size
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=2
        )

        anomaly_flags = []
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            nll = compute_nll_loss(model, batch)
            pred_nll = nll[:, args.context_len:, :]
            point_nll = pred_nll.mean(dim=-1)  # (B, PRED)

            point_anomalies = (point_nll > thr_nll).int()
            bad_count = point_anomalies.sum(dim=1)
            total_points = point_nll.shape[1]
            sample_anomaly = (bad_count > total_points * args.vote_rate).int()
            anomaly_flags.append(sample_anomaly.cpu().numpy())

        if not anomaly_flags:
            print(f"  ⚠️ No samples in {csv_file}, skip.")
            continue

        anomaly_flags = np.concatenate(anomaly_flags)
        anomaly_rate = anomaly_flags.mean()
        print(f"  Samples: {len(anomaly_flags)} | Anomaly Rate: {anomaly_rate*100:.2f}%")

        result_file.write(f"File: {csv_file}, Rate: {anomaly_rate:.6f}\n")
        result_file.flush()
        np.save(os.path.join(out_root, f"{csv_file}_flags.npy"), anomaly_flags)

    result_file.close()
    print(f"\n✅ All results saved to {out_root}")


# ===============================
#           参数解析
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moirai Anomaly Detection")

    # 模型与路径
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to finetuned Moirai model dir (e.g., ./checkpoints/moirai-finetuned)')
    parser.add_argument('--test_root', type=str, required=True,
                        help='Path to test folder containing CSV files (e.g., .../test/)')
    parser.add_argument('--setting', type=str, default='moirai_anomaly_test',
                        help='Experiment name for saving results')

    # 数据参数（必须与训练时一致！）
    parser.add_argument('--context_len', type=int, default=256)
    parser.add_argument('--pred_len', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)

    # 异常检测参数
    parser.add_argument('--percentile', type=float, default=99.9,
                        help='Percentile for NLL threshold (e.g., 99.9)')
    parser.add_argument('--vote_rate', type=float, default=0.01,
                        help='If > vote_rate of points are anomalous, mark sample as anomalous')

    # GPU
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    main(args)