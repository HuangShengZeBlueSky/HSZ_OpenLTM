import os
from modelscope.hub.snapshot_download import snapshot_download

# === 1. 定义存储位置 ===
MODEL_ROOT = "/home/ubuntu/hsz/models"
if not os.path.exists(MODEL_ROOT):
    os.makedirs(MODEL_ROOT)

print(f"准备下载模型到: {MODEL_ROOT} ...")

# === 2. 下载 Llama-2-7b (修正了 Repo ID) ===
# 旧ID (404): modelscope/Llama-2-7b-hf
# 新ID (可用): LLM-Research/meta-llama-Llama-2-7b-hf
print("\n>>> Downloading Llama-2-7b ...")
try:
    llama_path = snapshot_download('LLM-Research/meta-llama-Llama-2-7b-hf', cache_dir=MODEL_ROOT)
    print(f"Llama-2 下载完成: {llama_path}")
except Exception as e:
    print(f"Llama-2 下载失败: {e}")

# === 3. 下载 OPT-125m (使用 HF-Mirror 社区源) ===
print("\n>>> Downloading OPT-125m ...")
try:
    # 尝试使用 HF 镜像源的 OPT
    opt_path = snapshot_download('HF1s/opt-125m', cache_dir=MODEL_ROOT)
    print(f"OPT 下载完成: {opt_path}")
except Exception as e:
    print(f"OPT 下载失败: {e}")

# === 4. 下载 GPT2 (使用 AI-ModelScope 或 huggingface 镜像) ===
print("\n>>> Downloading GPT2 ...")
try:
    # GPT2 比较通用，尝试这个 ID
    gpt2_path = snapshot_download('AI-ModelScope/gpt2', cache_dir=MODEL_ROOT)
    print(f"GPT2 下载完成: {gpt2_path}")
except Exception as e:
    print(f"GPT2 下载失败: {e}")

print("\nAll Done! 请根据上面显示的'下载完成'路径，去修改 autotimes.py")