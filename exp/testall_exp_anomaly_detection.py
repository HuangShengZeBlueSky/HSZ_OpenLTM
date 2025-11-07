import os
import time
import warnings
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import json
from typing import Tuple

warnings.filterwarnings('ignore')

# ===== 异常检测辅助函数（自适应通道数） =====
EPS = 1e-8

def _mad_flags(residuals: np.ndarray, alpha: float = 3.5) -> np.ndarray:
    """
    residuals: (M, C) 非负残差矩阵（|y - y_hat|）
    返回布尔矩阵 (M, C)，True 表示该列该行异常
    """
    med = np.median(residuals, axis=0, keepdims=True)
    mad = np.median(np.abs(residuals - med), axis=0, keepdims=True) + EPS
    robust_z = np.abs(residuals - med) / (1.4826 * mad)
    return robust_z > alpha

def _quantile_flags(residuals: np.ndarray, q: float = 0.995) -> np.ndarray:
    thr = np.quantile(residuals, q, axis=0, keepdims=True)
    return residuals > thr

def _compute_anomaly_rate(preds: np.ndarray,
                          trues: np.ndarray,
                          method: str = "mad",
                          alpha: float = 3.5,
                          q: float = 0.995,
                          only_idx: int = -1) -> Tuple[float, np.ndarray]:
    """
    计算样本级异常率（N 级），并返回每个样本是否异常的布尔向量。
    - preds/trues: 形状 (N, T, C) 或 (N, T)；自动转为 (N, T, C)
    - method: "mad" 或 "quantile"
    - only_idx: 仅针对某一通道做检测（-1 表示全部通道）
    """
    # 统一形状到 (N, T, C)
    if preds.ndim == 2:
        preds = preds[..., None]
        trues = trues[..., None]
    assert preds.shape == trues.shape and preds.ndim == 3, "preds/trues 形状必须一致，且为 (N,T,C) 或 (N,T)"
    N, T, C = preds.shape

    # 计算残差并展平到 (M, C)
    residuals = np.abs(trues - preds)  # (N, T, C)
    if only_idx >= 0:
        residuals = residuals[:, :, [only_idx]]
        C = 1
    residuals = residuals.reshape(N * T, C)

    # 逐列检测并聚合
    if method == "mad":
        flags = _mad_flags(residuals, alpha=alpha)        # (M, C)
    else:
        flags = _quantile_flags(residuals, q=q)           # (M, C)
    row_flags = flags.any(axis=1)                         # (M,)
    row_flags = row_flags.reshape(N, T)                   # (N, T)
    sample_flags = row_flags.any(axis=1)                  # (N,)
    anomaly_rate = float(sample_flags.mean())
    return anomaly_rate, sample_flags


class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)
        
    def _build_model(self):
        if self.args.ddp:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            # for methods that do not use ddp (e.g. finetuning-based LLM4TS models)
            self.device = self.args.gpu
        
        model = self.model_dict[self.args.model].Model(self.args)
        
        if self.args.ddp:
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        elif self.args.dp:
            model = DataParallel(model, device_ids=self.args.device_ids).to(self.device)
        else:
            self.device = self.args.gpu
            model = model.to(self.device)
            
        if self.args.adaptation:
            model.load_state_dict(torch.load(self.args.pretrain_model_path))
        return model

    # <<< 修改点 1：为 _get_data 添加 test_data_path 参数 >>>
    def _get_data(self, flag, test_data_path=None):
        # 你的 data_provider 需要支持 test_data_path 参数
        data_set, data_loader = data_provider(self.args, flag, test_data_path)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        
        self.model.eval()    
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                if is_test or self.args.nonautoregressive:
                        outputs = outputs[:, -self.args.output_token_len:, :]
                        batch_y = batch_y[:, -self.args.output_token_len:, :].to(self.device)
                else:
                    outputs = outputs[:, :, :]
                    batch_y = batch_y[:, :, :].to(self.device)

                if self.args.covariate:
                    if self.args.last_token:
                        outputs = outputs[:, -self.args.output_token_len:, -1]
                        batch_y = batch_y[:, -self.args.output_token_len:, -1]
                    else:
                        outputs = outputs[:, :, -1]
                        batch_y = batch_y[:, :, -1]
                loss = criterion(outputs, batch_y)

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
        if self.args.ddp:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
            
        if self.args.model == 'gpt4ts':
            # GPT4TS just requires to train partial layers
            self.model.in_layer.train()
            self.model.out_layer.train()
        else: 
            self.model.train()
            
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                if self.args.dp:
                    torch.cuda.synchronize()
                if self.args.nonautoregressive:
                    batch_y = batch_y[:, -self.args.output_token_len:, :]
                if self.args.covariate:
                    if self.args.last_token:
                        outputs = outputs[:, -self.args.output_token_len:, -1]
                        batch_y = batch_y[:, -self.args.output_token_len:, -1]
                    else:
                        outputs = outputs[:, :, -1]
                        batch_y = batch_y[:, :, -1]
                loss = criterion(outputs, batch_y)
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                loss.backward()
                model_optim.step()

            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            vali_loss = self.vali(vali_data, vali_loader, criterion, is_test=self.args.valid_last)
            test_loss = self.vali(test_data, test_loader, criterion, is_test=True)
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {}, Steps: {} | Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                    epoch + 1, train_steps, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.ddp:
                train_loader.sampler.set_epoch(epoch + 1)
                
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.ddp:
            dist.barrier()
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    # <<< 修改点 2：重写 test 函数以支持遍历文件夹 >>>
    def test(self, setting, test=0):
        
        # 1. 加载模型
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

        # 2. 准备输出文件夹
        out_root = './test_results/' + setting + '/'
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        
        # 3. 查找所有要测试的 .csv 文件
        # [注意]：我们假设 --root_path 指向包含 .csv 文件的文件夹
        test_folder = self.args.root_path
        csv_files = sorted([f for f in os.listdir(test_folder) if f.endswith(".csv")])
        
        if not csv_files:
            print(f"警告：在 {test_folder} 中未找到任何 .csv 文件。")
            # 即使没有文件，也尝试加载 --data_path (如果它不是空的)
            if self.args.data_path:
                 print(f"回退到使用 --data_path: {self.args.data_path}")
                 csv_files = [self.args.data_path]
            else:
                 return # 找不到文件，提前退出

        # 准备一个文件来保存所有文件的汇总结果
        all_results_file = open(os.path.join(out_root, "result_summary.txt"), 'a')

        # 4. === 开始遍历每个 .csv 文件 ===
        for csv_file in csv_files:
            fp = os.path.join(test_folder, csv_file)
            print(f"\n>>>>>>> 正在处理 (Processing): {fp} <<<<<<<")

            # 为这个文件专门获取数据加载器
            _, test_loader = self._get_data(flag='test', test_data_path=fp)

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
                    
                    # (自回归预测逻辑... 与原来保持一致)
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
                    
                    # (可视化逻辑... 与原来保持一致)
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
                anomaly_rate, sample_flags = _compute_anomaly_rate(
                    preds=preds, trues=trues,
                    method=anom_method, alpha=anom_alpha, q=anom_q, only_idx=only_idx
                )
                print(f">>> [{csv_file}] 异常检测 (Anomaly) | anomaly_rate(sample-level): {anomaly_rate:.6f} "
                      f"(method={anom_method}, alpha={anom_alpha}, q={anom_q}, only_idx={only_idx})")
                
                # 8. --- (文件级别) 保存该文件的结果 ---
                
                # a. 保存到汇总文件
                all_results_file.write(f"File: {csv_file}\n")
                all_results_file.write(f"  Forecast: mse:{mse:.4f}, mae:{mae:.4f}\n")
                all_results_file.write(f"  Anomaly: rate:{anomaly_rate:.6f}\n\n")
                all_results_file.flush() # 立即写入
                
                # b. （可选）为该文件保存单独的异常标签
                np.save(os.path.join(out_root, f"{csv_file}_anomaly_flags.npy"), sample_flags)

            except Exception as e:
                print(f"[{csv_file}] 异常检测失败: {e}")
        
        # === 所有文件处理完毕 ===
        all_results_file.close()
        print("\n>>>>>>> 所有测试文件处理完毕 (All files processed.) <<<<<<<")
        return