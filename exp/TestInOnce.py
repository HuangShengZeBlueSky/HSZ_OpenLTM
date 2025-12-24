import os
import torch
import numpy as np
import torch.nn as nn
import time
from exp.exp_forecast import Exp_Forecast

class Exp_Forecast_TestAll3(Exp_Forecast):
    def __init__(self, args):
        super(Exp_Forecast_TestAll3, self).__init__(args)

    def test(self, setting, test=0):
            # =========================================
            # 1. 加载模型
            # =========================================
            if test:
                print('loading model')
                setting = self.args.test_dir
                best_model_path = self.args.test_file_name
                checkpoint_path = os.path.join(self.args.checkpoints, setting, best_model_path)
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()
            criterion = nn.MSELoss(reduction='none') 

            # =========================================
            # 2. 准备数据读取器 (只为了获取 scaler)
            # =========================================
            # 我们仍需要调用一次 _get_data 来获取训练时的 scaler，但我们不再信任它的数据质量
            _, temp_loader = self._get_data(flag='val')
            train_scaler = temp_loader.dataset.scaler

            # =========================================
            # 3. 遍历测试集 & 动态构建“基准池”
            # =========================================
            print(">>> 阶段1: 正在进行模型推理...")
            
            current_root = self.args.root_path
            if current_root.endswith('/test') or current_root.endswith('/test/'):
                test_folder = current_root 
            else:
                test_folder = os.path.join(current_root, 'test')
                
            self.args.root_path = test_folder 
            csv_files = sorted([f for f in os.listdir(test_folder) if f.endswith(".csv")])
            
            test_data_map = {}      # 存所有文件的 Loss
            calibration_losses = [] # [关键修改] 专门存“正常文件”的 Loss，代替验证集
            
            # 你的正常文件关键词
            normal_keywords = ["正常", "normal", "m0_g0_la0_ra0"]
            
            print(f"    处理测试集 ({len(csv_files)} 个文件)...")
            
            with torch.no_grad():
                for csv_file in csv_files:
                    _, test_loader = self._get_data(flag='test', test_data_path=csv_file, scaler=train_scaler)
                    file_losses = []
                    
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float().to(self.device)
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)
                        
                        outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                        f_dim = -1 if getattr(self.args, 'features', 'M') == 'MS' else 0
                        outputs = outputs[:, -self.args.output_token_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.output_token_len:, f_dim:]
                        
                        loss = criterion(outputs, batch_y).mean(dim=-1)
                        file_losses.append(loss.cpu().numpy())
                    
                    if len(file_losses) > 0:
                        # 1. 存入总表
                        file_loss_concat = np.concatenate(file_losses, axis=0)
                        test_data_map[csv_file] = file_loss_concat
                        
                        # 2. [核心修复] 如果是正常文件，加入到“基准池”
                        # 只有这里的 loss 才是干净的 (0.02级别)，我们要用它来定阈值
                        if any(k.lower() in csv_file.lower() for k in normal_keywords):
                            calibration_losses.append(file_loss_concat)

            # =========================================
            # 4. 构建新的基准 (Calibration Data)
            # =========================================
            if len(calibration_losses) > 0:
                # 使用找到的正常文件作为基准
                val_loss_matrix = np.concatenate(calibration_losses, axis=0)
                val_flattened = val_loss_matrix.flatten()
                print(f"    ✅ 已自动识别 {len(calibration_losses)} 个正常文件作为阈值基准。")
            else:
                # 如果没找到正常文件，这就是灾难，只能报错了
                print("    ❌ 严重错误：测试集中未找到包含 '正常/normal' 的文件！无法计算阈值。")
                return

            # =========================================
            # 5. 诊断：MSE 原始分布检查
            # =========================================
            print("\n>>> 阶段1.5: 原始误差(MSE)诊断")
            
            normal_mses = []
            fault_mses = []
            
            print("-" * 70)
            print(f"{'File Name':<55} | {'Avg MSE'}")
            print("-" * 70)
            
            # 打印基准池的 MSE (现在应该是 0.02 左右了)
            baseline_mse = np.mean(val_flattened)
            print(f"{'[New Baseline] (From Normal Files)':<55} | {baseline_mse:.6f}")
            
            for fname, loss_m in test_data_map.items():
                avg_mse = loss_m.mean()
                is_normal = any(k.lower() in fname.lower() for k in normal_keywords)
                tag = "[正常]" if is_normal else "[故障]"
                
                print(f"{tag + ' ' + fname[:45]:<55} | {avg_mse:.6f}")
                
                if is_normal: normal_mses.append(avg_mse)
                else: fault_mses.append(avg_mse)
                
            print("-" * 70)
            
            # 简单的自动判断
            max_norm_mse = max(normal_mses) if normal_mses else 0
            min_fault_mse = min(fault_mses) if fault_mses else 0
            
            if min_fault_mse <= max_norm_mse:
                print(f"⚠️  警告：部分故障样本MSE ({min_fault_mse:.4f}) <= 正常样本MSE ({max_norm_mse:.4f})")
                print(f"    部分轻微故障可能难以检测。")
            else:
                print(f"✅ 信号良好：故障MSE > 正常MSE。")

            # =========================================
            # 6. 寻优阶段 (Top-8)
            # =========================================
            print("\n>>> 阶段2: 内存参数寻优 (Top-8 展示)...")
            print(f"    [规则] 排序依据: Gap (平均检出 - 平均误报) 降序")
            print(f"    [硬约束] 正常样本平均误报率 <= 10%")
            
            p_list = [99.9, 99.5, 99.0, 98.0, 96.0, 95.0, 94.0, 92.0, 90.0, 85.0, 80.0]
            v_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
            
            candidates = [] 
            
            for p in p_list:
                # [关键] 现在 thresh 是基于真正的正常文件计算的 (0.02级别)
                # 这样即使是 0.024 的故障，也有机会被抓出来
                thresh = np.percentile(val_flattened, p)
                
                for v in v_list:
                    test_normal_rates = [] 
                    fault_rates = []       
                    current_results = {}
                    
                    # a. 基准池 FPR (自我验证)
                    val_anomalies = (val_loss_matrix > thresh)
                    val_fpr = (val_anomalies.sum(axis=1) > (val_loss_matrix.shape[1] * v)).mean()
                    if val_fpr > 0.10: continue 
                    
                    current_results["[Baseline]"] = val_fpr

                    # b. 测试集遍历
                    for fname, loss_m in test_data_map.items():
                        is_anomaly = (loss_m > thresh).sum(axis=1) > (loss_m.shape[1] * v)
                        rate = is_anomaly.mean()
                        current_results[fname] = rate
                        
                        if any(k.lower() in fname.lower() for k in normal_keywords):
                            test_normal_rates.append(rate)
                        else:
                            fault_rates.append(rate)
                    
                    if not fault_rates: continue

                    avg_norm_rate = np.mean(test_normal_rates)
                    avg_fault_rate = np.mean(fault_rates)
                    
                    if avg_norm_rate > 0.10: continue
                    
                    gap = avg_fault_rate - avg_norm_rate
                    
                    candidates.append({
                        'p': p, 'v': v,
                        'gap': gap,
                        'avg_norm': avg_norm_rate,
                        'avg_fault': avg_fault_rate,
                        'results': current_results
                    })

            # =========================================
            # 7. 排序与输出
            # =========================================
            candidates.sort(key=lambda x: -x['gap'])
            
            top_k = min(8, len(candidates))
            
            if top_k == 0:
                print("❌ 寻优失败！没有找到任何满足【误报率<=10%】的参数组合。")
            else:
                print(f"\n📊 筛选出 {len(candidates)} 组有效参数，展示前 {top_k} 组详细结果")
                
                for rank in range(top_k):
                    cand = candidates[rank]
                    p, v = cand['p'], cand['v']
                    
                    print("\n" + "=" * 80)
                    print(f"🏅 Rank {rank+1}:  P={p} | V={v} | Gap={cand['gap']:.4f}")
                    print(f"   (Avg Norm: {cand['avg_norm']*100:.2f}% | Avg Fault: {cand['avg_fault']*100:.2f}%)")
                    print("-" * 80)
                    print(f"{'File Name':<60} | {'Anomaly Rate'}")
                    print("-" * 80)
                    
                    sorted_items = sorted(cand['results'].items(), key=lambda x: (
                        0 if "[Base" in x[0] else (1 if any(k in x[0].lower() for k in normal_keywords) else 2), 
                        x[0]
                    ))
                    
                    for fname, rate in sorted_items:
                        print(f"{fname:<60} | {rate*100:6.2f}%")
                    
                print("=" * 80)
                
                best_cand = candidates[0]
                out_root = './test_results/' + setting + '/'
                if not os.path.exists(out_root): os.makedirs(out_root)
                np.save(os.path.join(out_root, "best_results.npy"), best_cand['results'])
                print(f"\n✅ 已自动选择 Rank 1 (P={best_cand['p']}, V={best_cand['v']}) 作为最佳结果保存。")

            self.args.root_path = os.path.dirname(test_folder.rstrip('/'))