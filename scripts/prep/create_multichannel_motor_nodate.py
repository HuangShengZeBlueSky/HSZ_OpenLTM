import pandas as pd
import os

# 1. 设置目标文件夹路径
folder_path = "/home/ubuntu/hsz/OpenLTM_data_backup/datasets/中车研究院实验台数据集/工况2_转速14转每秒/test"
# 检查文件夹是否存在
if not os.path.exists(folder_path):
    print(f"错误：找不到文件夹 {folder_path}")
    exit()

# 2. 获取所有 csv 文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
total_files = len(csv_files)

print(f">>> 找到 {total_files} 个 CSV 文件，准备开始处理...\n")

success_count = 0
fail_count = 0

for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)
    
    try:
        # 读取数据 (假设无表头，直接读取所有行)
        df = pd.read_csv(file_path, header=None)
        
        # 简单的跳过检查：如果已经是偶数项列（比如2列），可能已经处理过了？
        # 为了安全，这里只打印一下形状，但依然按照你的要求执行复制
        # 如果你想防止重复运行，可以取消下面两行的注释:
        # if df.shape[1] >= 2:
        #     print(f"跳过 {file_name}: 已经是 {df.shape[1]} 列了")
        #     continue

        # 核心操作：复制自身，拼接成双倍列 (1列变2列)
        df_new = pd.concat([df, df], axis=1)
        
        # 覆盖保存 (不保留索引和表头)
        df_new.to_csv(file_path, header=False, index=False)
        
        print(f"✅ 已处理: {file_name} | 形状变化: {df.shape} -> {df_new.shape}")
        success_count += 1
        
    except Exception as e:
        print(f"❌ 处理失败: {file_name} | 原因: {e}")
        fail_count += 1

print("-" * 30)
print(f"处理完成！成功: {success_count}, 失败: {fail_count}")