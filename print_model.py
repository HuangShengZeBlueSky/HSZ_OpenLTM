import torch

# 替换成你的 13MB 文件路径
path_13mb = '/home/ubuntu/hsz/OpenLTM/checkpoints/forecast_motor_mv_nodate_timer_MultivariateDatasetBenchmark_sl32_it32_ot8_lr0.0001_bt16_wd0_el1_dm512_dff2048_nh8_cosFalse_test_0/checkpoint.pth'

# 1. 加载文件
try:
    weights_13mb = torch.load(path_13mb, map_location='cpu')

    # 2. 检查类型
    if isinstance(weights_13mb, dict):
        print("类型: 这是一个字典 (state_dict)。")
        
        # 3. 打印一些键来预览
        keys_13mb = list(weights_13mb.keys())
        print(f"总共有 {len(keys_13mb)} 个权重张量。")
        print("\n前 5 个权重键 (Key) 示例:")
        for key in keys_13mb[:5]:
            # 打印键名 和 对应张量的形状
            print(f"  - {key}: \t{weights_13mb[key].shape}")
            
        print("\n后 5 个权重键 (Key) 示例:")
        for key in keys_13mb[-5:]:
            print(f"  - {key}: \t{weights_13mb[key].shape}")

    else:
        print(f"类型: {type(weights_13mb)} (不是一个 state_dict 字典!)")

except Exception as e:
    print(f"加载 13MB 文件失败: {e}")