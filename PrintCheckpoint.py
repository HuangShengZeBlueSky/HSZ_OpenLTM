import torch
import os

# 你的 Checkpoint 路径
ckpt_path = "/raid/hsz/HSZ_OpenLTM/checkpoints/forecast_工况1_转速12转每秒_微调_autotimes_BJTU_sl2880_it96_ot96_lr0.0001_bt256_wd0_el8_dm1024_dff2048_nh8_cosFalse_test_0/checkpoint.pth"

if not os.path.exists(ckpt_path):
    print(f"错误: 找不到文件 {ckpt_path}")
else:
    print(f"正在加载 Checkpoint: {ckpt_path} ...\n")
    # 加载到 CPU 避免爆显存
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # 打印表头
    print(f"{'Key Name (层名称)':<40} | {'Shape (维度: 输出x输入)':<25} | {'Params (参数量)'}")
    print("-" * 85)

    total_params = 0
    lm_head_params = 0
    adapter_params = 0

    for key, value in state_dict.items():
        shape_str = str(list(value.shape))
        num_params = value.numel()
        total_params += num_params
        
        # 分类统计
        if "lm_head" in key:
            lm_head_params += num_params
            tag = " [无用垃圾]"
        else:
            adapter_params += num_params
            tag = " [有效权重]"

        print(f"{key:<40} | {shape_str:<25} | {num_params:,}{tag}")

    print("-" * 85)
    print(f"总参数量: {total_params:,}")
    print(f"  - 其中 'lm_head' (垃圾) 参数量: {lm_head_params:,} (约 {lm_head_params*4/1024/1024:.1f} MB)")
    print(f"  - 其中 Encoder/Decoder (有效) 参数量: {adapter_params:,} (约 {adapter_params*4/1024/1024:.1f} MB)")
    print("-" * 85)