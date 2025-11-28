# exp/exp_AnomalyDetection.py
import os
import torch
import numpy as np
import torch.nn as nn
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
        # [修改] reduction='none' 保留 (Batch, Time, Vars) 的形状
        criterion = nn.MSELoss(reduction='none') 

        # ===========================
        # 阶段 1: 验证集校准 (制定点级阈值)
        # ===========================
        print(">>> 开始校准 (Calibration)...")
        
        original_root = self.args.root_path
        if original_root.endswith('/test') or original_root.endswith('/test/'):
             self.args.root_path = os.path.dirname(original_root.rstrip('/'))
             
        _, val_loader = self._get_data(flag='val')
        print(f"  验证集样本数: {len(val_loader.dataset)}")
        
        train_scaler = val_loader.dataset.scaler
        
        val_point_losses = [] # 用于存储所有点的误差
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                
                f_dim = -1 if getattr(self.args, 'features', 'M') == 'MS' else 0
                outputs = outputs[:, -self.args.output_token_len:, f_dim:]
                batch_y = batch_y[:, -self.args.output_token_len:, f_dim:]
                
                # [核心修改] 
                # loss shape: (Batch, Time, Vars)
                loss = criterion(outputs, batch_y)
                
                # 我们只关心时间点维度的误差，所以先把变量维度(Vars)平均掉
                # point_loss shape: (Batch, Time)
                point_loss = loss.mean(dim=-1) 
                
                # 展平放入列表，因为我们要算所有点的分位数
                val_point_losses.append(point_loss.cpu().numpy().flatten())

        # 将验证集所有样本、所有时间点的误差拼在一起
        val_point_losses = np.concatenate(val_point_losses, axis=0)
        
        # [核心修改] 计算“点级阈值”
        # 比如 99.9 分位数是 2.5，意味着只有 0.1% 的点误差会超过 2.5
        thr_point = np.percentile(val_point_losses, percentile)
        print(f"  点级阈值 ({percentile}%): {thr_point:.6f}")

        # ===========================
        # 阶段 2: 测试集遍历 (投票机制)
        # ===========================
        test_folder = os.path.join(self.args.root_path, 'test')
        self.args.root_path = test_folder 
        
        csv_files = sorted([f for f in os.listdir(test_folder) if f.endswith(".csv")])
        
        # 定义投票比例：比如 1% (0.01) 的点异常，则判定该样本异常
        # [修改] 使用 args.vote_rate
        vote_rate = getattr(self.args, 'vote_rate', 0.01)
        print(f"  策略配置: 阈值分位数={percentile}%, 异常点投票率={vote_rate*100}%")

        
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

                    # loss shape: (Batch, Time, Vars)
                    loss = criterion(outputs, batch_y)
                    
                    # 1. 得到每个点的误差 (Batch, Time)
                    point_loss = loss.mean(dim=-1) 
                    
                    # 2. 判断每个点是否异常 (0 或 1)
                    # point_anomalies shape: (Batch, Time)
                    point_anomalies = (point_loss > thr_point).int()
                    
                    # 3. 统计每个样本中有多少个异常点 (Batch, )
                    bad_points_count = point_anomalies.sum(dim=1)
                    
                    # 4. 投票判定：如果 (异常点数 / 总点数) > 0.01，则该样本异常
                    total_points = point_loss.shape[1] # 通常是 96
                    sample_is_anomaly = (bad_points_count > (total_points * vote_rate)).int()
                    
                    anomaly_flags.append(sample_is_anomaly.cpu().numpy())

            if len(anomaly_flags) == 0: continue

            anomaly_flags = np.concatenate(anomaly_flags, axis=0)
            anomaly_rate = np.mean(anomaly_flags)

            print(f"  Samples: {len(anomaly_flags)} | Rate: {anomaly_rate*100:.2f}%")
            
            all_results_file.write(f"File: {csv_file}, Rate: {anomaly_rate:.4f}\n")
            all_results_file.flush()
            np.save(os.path.join(out_root, f"{csv_file}_flags.npy"), anomaly_flags)

        all_results_file.close()
        self.args.root_path = original_root 
        return