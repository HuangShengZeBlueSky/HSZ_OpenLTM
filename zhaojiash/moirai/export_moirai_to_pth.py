# convert_moirai_safetensors_to_pth.py
import torch
from safetensors import safe_open
import os
import json

# === 配置路径 ===
SAFETENSORS_PATH = "/home/ubuntu/zhaojia/checkpoints/Moirai/model.safetensors"
CONFIG_PATH = "/home/ubuntu/zhaojia/checkpoints/Moirai/config.json"          # 可选，用于验证
OUTPUT_PTH_PATH = "/home/ubuntu/zhaojia/checkpoints/Moirai/moirai-1.1-R-base.pth"

# === 加载 config（可选）===
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    print("🔍 模型配置:")
    print(f"   d_model: {config.get('d_model', 'N/A')}")
    print(f"   num_layers: {config.get('num_layers', 'N/A')}")

# === 加载 safetensors 权重 ===
print("\n📦 正在加载 model.safetensors ...")
state_dict = {}
with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)

print(f"✅ 共加载 {len(state_dict)} 个参数")

# === 提取 backbone 并移除前缀 ===
# Moirai 的完整模型结构是：MoiraiForPrediction(backbone=PatchTSMixer...)
# OpenLTM 微调时通常只需要 backbone 部分，且不带 'backbone.' 前缀
backbone_state = {}
for k, v in state_dict.items():
    if k.startswith("backbone."):
        new_k = k[len("backbone."):]  # 移除 'backbone.' 前缀
        backbone_state[new_k] = v

print(f"✂️  提取 backbone 参数: {len(backbone_state)} 个")

# === 保存为 .pth ===
torch.save(backbone_state, OUTPUT_PTH_PATH)
print(f"\n🎉 成功保存 Moirai backbone 权重到:\n   {OUTPUT_PTH_PATH}")