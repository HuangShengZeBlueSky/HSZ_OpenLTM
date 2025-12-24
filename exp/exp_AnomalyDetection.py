import os
import torch
import numpy as np
import torch.nn as nn
import time
from exp.exp_forecast import Exp_Forecast

class Exp_Forecast_TestAll2(Exp_Forecast):
    def __init__(self, args):
        super(Exp_Forecast_TestAll2, self).__init__(args)

    def test(self, setting, test=0):
        # 1. 加载模型
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            checkpoint_path = os.path.join(self.args.checkpoints, setting, best_model_path)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)
            print(f">> 忽略的 OPT 参数数: {len(missing)}, 额外键: {unexpected}")
        self.model.eval()
        
        out_root = './test_results/' + setting + '/'
        if not os.path.exists(out_root): os.makedirs(out_root)
        all_results_file = open(os.path.join(out_root, "result_summary.txt"), 'a')
        
        percentile = getattr(self.args, 'percentile', 99.9)
        criterion = nn.MSELoss(reduction='none') 

        # ===========================
        # 阶段 1: 验证集校准 (Calibration) - [优化版: 降采样]
        # ===========================
        print(">>> 开始校准 (Calibration)...")
        t_cali_start = time.time() # 计时
        
        original_root = self.args.root_path
        if original_root.endswith('/test') or original_root.endswith('/test/'):
             self.args.root_path = os.path.dirname(original_root.rstrip('/'))
             
        _, val_loader = self._get_data(flag='val')
        total_val_samples = len(val_loader)
        print(f"    验证集总 Batch 数: {total_val_samples}")
        
        train_scaler = val_loader.dataset.scaler
        val_point_losses = [] 
        
        # [配置] 降采样比例：0.1 表示只跑 10% 的验证集数据用于计算阈值
        # 10万样本 -> 1万样本，速度提升10倍，统计误差通常可忽略
        cali_ratio = 0.1 
        step = int(1 / cali_ratio) if cali_ratio < 1.0 else 1
        print(f"    降采样策略: 仅使用 {cali_ratio*100}% 的验证集数据 (每 {step} 个Batch取 1 个)")

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
                # [核心优化] 降采样跳过
                if i % step != 0:
                    continue

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                
                f_dim = -1 if getattr(self.args, 'features', 'M') == 'MS' else 0
                outputs = outputs[:, -self.args.output_token_len:, f_dim:]
                batch_y = batch_y[:, -self.args.output_token_len:, f_dim:]
                
                loss = criterion(outputs, batch_y)
                point_loss = loss.mean(dim=-1) 
                val_point_losses.append(point_loss.cpu().numpy().flatten())

        val_point_losses = np.concatenate(val_point_losses, axis=0)
        thr_point = np.percentile(val_point_losses, percentile)
        
        t_cali_end = time.time()
        print(f"    校准耗时: {t_cali_end - t_cali_start:.2f}s")
        print(f"    实际计算点数: {len(val_point_losses)}")
        print(f"    点级阈值 ({percentile}%): {thr_point:.6f}")

        # ===========================
        # 阶段 2: 测试集遍历 (保持不变)
        # ===========================
        test_folder = os.path.join(self.args.root_path, 'test')
        self.args.root_path = test_folder 
        
        csv_files = sorted([f for f in os.listdir(test_folder) if f.endswith(".csv")])
        vote_rate = getattr(self.args, 'vote_rate', 0.01)
        
        print(f"    策略配置: 阈值分位数={percentile}%, 异常点投票率={vote_rate*100}%")

        for csv_file in csv_files:
            print(f">>> Testing: {csv_file}")
            _, test_loader = self._get_data(flag='test', test_data_path=csv_file, scaler=train_scaler)
            
            anomaly_flags = []
            
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                    
                    f_dim = -1 if getattr(self.args, 'features', 'M') == 'MS' else 0
                    outputs = outputs[:, -self.args.output_token_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.output_token_len:, f_dim:]

                    loss = criterion(outputs, batch_y)
                    point_loss = loss.mean(dim=-1) 
                    point_anomalies = (point_loss > thr_point).int()
                    bad_points_count = point_anomalies.sum(dim=1)
                    
                    total_points = point_loss.shape[1]
                    sample_is_anomaly = (bad_points_count > (total_points * vote_rate)).int()
                    anomaly_flags.append(sample_is_anomaly.cpu().numpy())

            if len(anomaly_flags) == 0: continue

            anomaly_flags = np.concatenate(anomaly_flags, axis=0)
            anomaly_rate = np.mean(anomaly_flags)

            print(f"    Samples: {len(anomaly_flags)} | Rate: {anomaly_rate*100:.2f}%")
            
            all_results_file.write(f"File: {csv_file}, Rate: {anomaly_rate:.4f}\n")
            all_results_file.flush()
            np.save(os.path.join(out_root, f"{csv_file}_flags.npy"), anomaly_flags)

        all_results_file.close()
        self.args.root_path = original_root 
        return