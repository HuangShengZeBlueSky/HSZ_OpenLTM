from data_provider.data_loader import UnivariateDatasetBenchmark, MultivariateDatasetBenchmark, Global_Temp, Global_Wind, Dataset_ERA5_Pretrain, Dataset_ERA5_Pretrain_Test, UTSD, UTSD_Npy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

data_dict = {
    'UnivariateDatasetBenchmark': UnivariateDatasetBenchmark,
    'MultivariateDatasetBenchmark': MultivariateDatasetBenchmark,
    'Global_Temp': Global_Temp,
    'Global_Wind': Global_Wind,
    'Era5_Pretrain': Dataset_ERA5_Pretrain,
    'Era5_Pretrain_Test': Dataset_ERA5_Pretrain_Test,
    'Utsd': UTSD,
    'Utsd_Npy': UTSD_Npy
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

def data_provider(args, flag):
    # 解析别名
    data_key = DATA_ALIASES.get(args.data, args.data)
    if data_key not in data_dict:
        raise ValueError(
            f"未知的数据集标识 args.data='{args.data}'. "
            f"可选值: {list(data_dict.keys())} 或别名: {list(DATA_ALIASES.keys())}"
        )
    Data = data_dict[data_key]

    if flag in ['test', 'val']:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size

    if flag in ['train', 'val']:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.input_token_len, args.output_token_len],
            nonautoregressive=args.nonautoregressive,
            test_flag=args.test_flag,
            subset_rand_ratio=args.subset_rand_ratio
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.test_seq_len, args.input_token_len, args.test_pred_len],
            nonautoregressive=args.nonautoregressive,
            test_flag=args.test_flag,
            subset_rand_ratio=args.subset_rand_ratio
        )
    print(flag, len(data_set))
    if args.ddp:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last
        )
    return data_set, data_loader
