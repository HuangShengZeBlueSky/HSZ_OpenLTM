import os
import time
import warnings
import torch
import numpy as np
import torch.nn as nn
from torch import optim

# 1. [!!] 导入你现有的、“干净”的 Exp_Forecast
from exp.exp_forecast import Exp_Forecast

# 2. [!!] 导入你现有的工具和指标
from utils.tools import visual
from utils.metrics import metric

# 3. [!!] 导入 Exp_Forecast 中的异常检测辅助函数
# (我们假设这些函数位于 exp_forecast.py 的顶部)
try:
    from exp.exp_forecast import _mad_flags, _quantile_flags, _compute_anomaly_rate
except ImportError:
    
    # 如果导入失败，在这里重新定义它们
    
    EPS = 1e-8
    
    def _mad_flags(residuals: np.ndarray, alpha: float = 3.5) -> np.ndarray:
        med = np.median(residuals, axis=0, keepdims=True)
        mad = np.median(np.abs(residuals - med), axis=0, keepdims=True) + EPS
        robust_z = np.abs(residuals - med) / (1.4826 * mad)
        return robust_z > alpha

    def _quantile_flags(residuals: np.ndarray, q: float = 0.995) -> np.ndarray:
        thr = np.quantile(residuals, q, axis=0, keepdims=True)
        return residuals > thr

    # [!!] 关键修复：这里是 _compute_anomaly_rate 的完整实现 [!!]
    def _compute_anomaly_rate(preds: np.ndarray, trues: np.ndarray, **kwargs) -> tuple:
        """
        这个实现从 kwargs 中提取参数，以匹配原始函数的签名。
        """
        # 1. 从 kwargs 中提取参数
        method = kwargs.get("method", "mad")
        alpha = kwargs.get("alpha", 3.5)
        q = kwargs.get("q", 0.995)
        only_idx = kwargs.get("only_idx", -1)

        # 2. 统一形状
        if preds.ndim == 2:
            preds = preds[..., None]
            trues = trues[..., None]
        
        if preds.shape != trues.shape or preds.ndim != 3:
             raise ValueError(f"preds/trues 形状必须一致且为 (N,T,C) 或 (N,T), 得到 {preds.shape} 和 {trues.shape}")

        N, T, C = preds.shape

        # 3. 计算残差并根据 only_idx 筛选
        residuals = np.abs(trues - preds)  # (N, T, C)
        if only_idx >= 0:
            if only_idx >= C:
                raise ValueError(f"only_idx {only_idx} 超出通道范围 {C}")
            residuals = residuals[:, :, [only_idx]]
            C = 1
        residuals = residuals.reshape(N * T, C)

        # 4. 逐列检测（使用 'method' 变量）
        if method == "mad":
            flags = _mad_flags(residuals, alpha=alpha)  # (M, C)
        elif method == "quantile":
            flags = _quantile_flags(residuals, q=q)     # (M, C)
        else:
            raise ValueError(f"未知的异常检测方法: {method}")
            
        # 5. 聚合
        row_flags = flags.any(axis=1)                   # (M,)
        row_flags = row_flags.reshape(N, T)             # (N, T)
        sample_flags = row_flags.any(axis=1)            # (N,)
        anomaly_rate = float(sample_flags.mean())
        
        return anomaly_rate, sample_flags


warnings.filterwarnings('ignore')


class Exp_Forecast_TestAll(Exp_Forecast):
    """
    这是一个专门用于“一键测试”的实验类。
    它继承了 Exp_Forecast 的所有功能（_build_model, _get_data, train, vali...）。
    它只重写（override）了 test 方法，赋予其遍历文件夹的能力。
    """
    def __init__(self, args):
        # 初始化父类 (Exp_Forecast)
        super(Exp_Forecast_TestAll, self).__init__(args)

    def test(self, setting, test=0):
        
        # 1. 加载模型 (与父类 Exp_Forecast 相同)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            checkpoint = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            for name, param in self.model.named_parameters():
                if not param.requires_grad and name not in checkpoint:
                    checkpoint[name] = param
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()

        # 2. 准备输出文件夹 (与父类 Exp_Forecast 相同)
        out_root = './test_results/' + setting + '/'
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        
        # 3. [!!] 遍历逻辑 [!!]
        test_folder = self.args.root_path
        csv_files = sorted([f for f in os.listdir(test_folder) if f.endswith(".csv")])
        
        if not csv_files:
            print(f"警告：在 {test_folder} 中未找到任何 .csv 文件。")
            if self.args.data_path and os.path.isfile(os.path.join(test_folder, self.args.data_path)):
                 print(f"回退到使用 --data_path: {self.args.data_path}")
                 csv_files = [self.args.data_path]
            else:
                 print(f"错误：在 {test_folder} 中找不到任何 .csv 文件，并且 --data_path ('{self.args.data_path}') 也无效。")
                 return

        all_results_file = open(os.path.join(out_root, "result_summary.txt"), 'a')
        
        # 保存原始 data_path，以便在循环后恢复
        original_data_path = self.args.data_path

        # 4. === 开始遍历每个 .csv 文件 ===
        for csv_file in csv_files:
            fp = os.path.join(test_folder, csv_file)
            print(f"\n>>>>>>> 正在处理 (Processing): {fp} <<<<<<<")

            # 临时修改 self.args 以便 _get_data 正确工作
            self.args.data_path = csv_file 
            
            try:
                # [!!] 调用父类的 _get_data
                _, test_loader = self._get_data(flag='test')
            except Exception as e:
                print(f"警告：为 {csv_file} 加载数据失败：{e}，跳过。")
                continue

            if not test_loader:
                print(f"警告：无法为 {csv_file} 加载数据，跳过。")
                continue

            preds = []
            trues = []
            time_now = time.time()
            test_steps = len(test_loader)
            iter_count = 0
            
            # 5. --- (内部循环) 遍历该文件的所有批次 ---
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    iter_count += 1
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    # (自回归预测逻辑... 与父类 Exp_Forecast 相同)
                    inference_steps = self.args.test_pred_len // self.args.output_token_len
                    dis = self.args.test_pred_len - inference_steps * self.args.output_token_len
                    if dis != 0:
                        inference_steps += 1
                    pred_y = []
                    for j in range(inference_steps):    
                        if len(pred_y) != 0:
                            batch_x = torch.cat([batch_x[:, self.args.input_token_len:, :], pred_y[-1]], dim=1)
                        outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                        pred_y.append(outputs[:, -self.args.output_token_len:, :])
                    pred_y = torch.cat(pred_y, dim=1)
                    if dis != 0:
                        pred_y = pred_y[:, :-self.args.output_token_len+dis, :]
                    batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)
                    
                    outputs = pred_y.detach().cpu()
                    batch_y = batch_y.detach().cpu()
                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)
                    
                    if (i + 1) % 100 == 0:
                        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                            speed = (time.time() - time_now) / iter_count
                            left_time = speed * (test_steps - i)
                            print("\t[文件: {0}] iters: {1}, speed: {2:.4f}s/iter, left time: {3:.4f}s".format(
                                csv_file, i + 1, speed, left_time))
                            iter_count = 0
                            time_now = time.time()
                    
                    if self.args.visualize and i % 2 == 0:
                        dir_path = out_root + f'{csv_file}_{self.args.test_pred_len}/'
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        gt = np.array(true[0, :, -1])
                        pd = np.array(pred[0, :, -1])
                        visual(gt, pd, os.path.join(dir_path, f'{i}.pdf'))
            
            # 6. --- (文件级别) 计算结果 ---
            if not preds:
                print(f"警告：文件 {csv_file} 没有生成任何预测结果，跳过。")
                continue

            preds = torch.cat(preds, dim=0).numpy()
            trues = torch.cat(trues, dim=0).numpy()
            print(f'[{csv_file}] preds shape: {preds.shape}')
            print(f'[{csv_file}] trues shape: {trues.shape}')
            
            if self.args.covariate:
                preds = preds[:, :, -1]
                trues = trues[:, :, -1]
                
            mae, mse, rmse, mape, mspe, smape = metric(preds, trues)
            print(f'>>> [{csv_file}] 预测结果 (Forecast) | mse:{mse:.4f}, mae:{mae:.4f}')

            # 7. --- (文件级别) 异常检测 ---
            anom_method = getattr(self.args, "anom_method", "mad")
            anom_alpha = getattr(self.args, "anom_alpha", 3.5)
            anom_q = getattr(self.args, "anom_q", 0.995)
            only_idx = getattr(self.args, "anom_only_idx", -1)

            try:
                # [!!] 现在这里会调用我们上面定义的、完整的 _compute_anomaly_rate
                anomaly_rate, sample_flags = _compute_anomaly_rate(
                    preds=preds, trues=trues,
                    method=anom_method, alpha=anom_alpha, q=anom_q, only_idx=only_idx
                )
                print(f">>> [{csv_file}] 异常检测 (Anomaly) | anomaly_rate(sample-level): {anomaly_rate:.6f} "
                      f"(method={anom_method}, alpha={anom_alpha}, q={anom_q}, only_idx={only_idx})")
                
                # 8. --- (文件级别) 保存该文件的结果 ---
                all_results_file.write(f"File: {csv_file}\n")
                all_results_file.write(f"  Forecast: mse:{mse:.4f}, mae:{mae:.4f}\n")
                all_results_file.write(f"  Anomaly: rate:{anomaly_rate:.6f}\n\n")
                all_results_file.flush()
                
                np.save(os.path.join(out_root, f"{csv_file}_anomaly_flags.npy"), sample_flags)

            except Exception as e:
                print(f"[{csv_file}] 异常检测失败: {e}")
        
        # === 所有文件处理完毕 ===
        all_results_file.close()
        self.args.data_path = original_data_path 
        print("\n>>>>>>> 所有测试文件处理完毕 (All files processed.) <<<<<<<")
        return