from data_provider.data_loader import UnivariateDatasetBenchmark, MultivariateDatasetBenchmark, Global_Temp, Global_Wind, Dataset_ERA5_Pretrain, Dataset_ERA5_Pretrain_Test, UTSD, UTSD_Npy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# [!!] 确保正确导入 BJTU 类
from data_provider.data_loader import BJTUAnomalyloader

data_dict = {
    'UnivariateDatasetBenchmark': UnivariateDatasetBenchmark,
    'MultivariateDatasetBenchmark': MultivariateDatasetBenchmark,
    'Global_Temp': Global_Temp,
    'Global_Wind': Global_Wind,
    'Era5_Pretrain': Dataset_ERA5_Pretrain,
    'Era5_Pretrain_Test': Dataset_ERA5_Pretrain_Test,
    'Utsd': UTSD,
    'Utsd_Npy': UTSD_Npy,
    'BJTU': BJTUAnomalyloader # [!!] 注册名 BJTU
}

# 新增：常见数据集名到内部数据类的映射（别名）
DATA_ALIASES = {
    # ETT/ECL 等常见表格数据，使用多变量基准数据类
    'ETTh1': 'MultivariateDatasetBenchmark',
    'ETTh2': 'MultivariateDatasetBenchmark',
    'ETTm1': 'MultivariateDatasetBenchmark',
    'ETTm2': 'MultivariateDatasetBenchmark',
    'ECL': 'MultivariateDatasetBenchmark',
    'Electricity': 'MultivariateDatasetBenchmark',
    'Weather': 'MultivariateDatasetBenchmark',
    'Traffic': 'MultivariateDatasetBenchmark',
}

def data_provider(args, flag, **kwargs):
    # 解析别名
    data_key = DATA_ALIASES.get(args.data, args.data)
    if data_key not in data_dict:
        raise ValueError(
            f"未知的数据集标识 args.data='{args.data}'. "
            f"可选值: {list(data_dict.keys())} 或别名: {list(DATA_ALIASES.keys())}"
        )
    Data = data_dict[data_key]

    # 安全获取 freq
    freq = getattr(args, 'freq', 'h')

    if flag in ['test', 'val']:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size

    if hasattr(args, 'task_name') and args.task_name == 'anomaly_detection':
        drop_last = False
        
    # [!!] BJTU 分支
    if data_key == 'BJTU':
        # [关键修改] 使用 pop 移除 key，防止重复传参
        target_test_path = kwargs.pop('test_data_path', args.data_path)

        data_set = Data(
            root_path=args.root_path,
            seq_len=args.seq_len,
            patch_len=args.input_token_len, 
            flag=flag,
            test_data_path=target_test_path,
            **kwargs 
        )
    
    # 标准分支
    else:
        timeenc = 0 if args.embed != 'timeF' else 1
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader