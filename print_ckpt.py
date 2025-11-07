import torch

# 替换成你的 29MB 文件路径
path_29mb = '/home/ubuntu/Timer/checkpoints/Timer_anomaly_detection_1.0.ckpt'

print(f"\n\n--- 正在检查 29MB 文件: {path_29mb} ---")

# 1. 加载文件
try:
    checkpoint_29mb = torch.load(path_29mb, map_location='cpu')

    # 2. 检查类型
    if isinstance(checkpoint_29mb, dict):
        print("类型: 这是一个字典 (Checkpoint)。")
        
        # 3. 打印顶层键 (Top-level Keys)
        keys_29mb = list(checkpoint_29mb.keys())
        print(f"\nCheckpoint 包含的顶层键 (Top-level Keys): {keys_29mb}")

        # --- 深入分析 ---

        # 4. 分析 'model' 或 'state_dict' (如果有)
        model_key = None
        if 'model' in keys_29mb:
            model_key = 'model'
        elif 'state_dict' in keys_29mb:
            model_key = 'state_dict'
        
        if model_key:
            print(f"\n[分析 '{model_key}' 键]:")
            model_weights = checkpoint_29mb[model_key]
            if isinstance(model_weights, dict):
                model_weight_keys = list(model_weights.keys())
                print(f"  -> '{model_key}' 是一个 state_dict，包含 {len(model_weight_keys)} 个权重。")
                print("  -> 前 5 个权重键示例:")
                for key in model_weight_keys[:5]:
                    print(f"     - {key}: \t{model_weights[key].shape}")
            else:
                 print(f"  -> '{model_key}' 的类型是 {type(model_weights)}。")

        # 5. 分析 'optimizer' (如果有)
        if 'optimizer' in keys_29mb:
            print(f"\n[分析 'optimizer' 键]:")
            optimizer_state = checkpoint_29mb['optimizer']
            if isinstance(optimizer_state, dict):
                print(f"  -> 'optimizer' 是一个字典，包含的键: {list(optimizer_state.keys())}")
                # 优化器状态通常很大，我们不打印它
            else:
                 print(f"  -> 'optimizer' 的类型是 {type(optimizer_state)}。")
                 
        # 6. 分析 'epoch' (如果有)
        if 'epoch' in keys_29mb:
             print(f"\n[分析 'epoch' 键]:")
             print(f"  -> Epoch: {checkpoint_29mb['epoch']}")

    else:
        print(f"类型: {type(checkpoint_29mb)} (不是一个 Checkpoint 字典!)")

except Exception as e:
    print(f"加载 29MB 文件失败: {e}")