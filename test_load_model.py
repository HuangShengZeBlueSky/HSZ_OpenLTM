import torch
from transformers import LlamaForCausalLM

# 这是你刚刚下载好的绝对路径
MODEL_PATH = "/home/ubuntu/hsz/models/NousResearch/Llama-2-7b-hf"

print(f"🚀 开始测试加载 Llama-2，路径: {MODEL_PATH}")
print("... 正在加载到内存 (预计消耗约 13GB 内存) ...")

try:
    # 关键点：这里我们强制使用 float16，模拟在 4090 上的显存占用情况
    model = LlamaForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    )
    print("✅ 加载成功！模型文件完整。")
    
    # 打印一下模型参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"📊 模型参数量: {params / 1e9:.2f} Billion")
    
except Exception as e:
    print(f"❌ 加载失败！错误信息:\n{e}")