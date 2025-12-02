# custom_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Optional, Tuple

# class MoiraiAnomalyDataset(Dataset):


#     def __init__(
#         self,
#         data_array: np.ndarray,  # 直接传入 numpy 数组，更灵活
#         context_length: int,
#         prediction_length: int,
#         use_single_variate: bool = True,
#         scaler: Optional[object] = None,
#     ):
#         if use_single_variate:
#             data = data_array[:, [0]].astype(np.float32)  # (T, 1)
#         else:
#             data = data_array.astype(np.float32)

#         self.context_length = context_length
#         self.prediction_length = prediction_length
#         self.total_length = context_length + prediction_length
#         self.D = data.shape[1]

#         if scaler is not None:
#             data = scaler.transform(data)
#         self.data = data

#         if len(self.data) < self.total_length:
#             raise ValueError(f"Data too short: {len(data)} < {self.total_length}")

#     def __len__(self):
#         return len(self.data) - self.total_length + 1

#     def __getitem__(self, idx):
#         window = self.data[idx : idx + self.total_length]
#         target = torch.from_numpy(window)

#         L = self.total_length
#         D = self.D

#         observed_mask = torch.ones(L, D, dtype=torch.bool)
#         time_id = torch.arange(L, dtype=torch.long).unsqueeze(1).expand(L, D)
#         variate_id = torch.arange(D, dtype=torch.long).unsqueeze(0).expand(L, D)
#         prediction_mask = torch.cat([
#             torch.zeros(self.context_length, D, dtype=torch.bool),
#             torch.ones(self.prediction_length, D, dtype=torch.bool)
#         ], dim=0)

#         return {
#             "target": target,
#             "observed_mask": observed_mask,
#             "time_id": time_id,
#             "variate_id": variate_id,
#             "prediction_mask": prediction_mask,
#         }
    


class MoiraiAnomalyDataset(Dataset):
    def __init__(self, data_array, context_length, prediction_length, use_single_variate=True):
        T, n_vars = data_array.shape
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.total_len = context_length + prediction_length

        if use_single_variate:
            self.samples = []
            for i in range(n_vars):
                series = data_array[:, i]  # (T,)
                for start in range(T - self.total_len + 1):
                    ctx = series[start:start+context_length]
                    tgt = series[start+context_length:start+self.total_len]
                    self.samples.append((ctx, tgt))
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctx, tgt = self.samples[idx]
        return {
            "past_values": torch.tensor(ctx, dtype=torch.float32).unsqueeze(0),
            "future_values": torch.tensor(tgt, dtype=torch.float32).unsqueeze(0),
            "past_observed_mask": torch.ones(len(ctx)),
            "future_observed_mask": torch.ones(len(tgt)),
        }