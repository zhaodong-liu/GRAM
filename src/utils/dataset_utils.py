import torch
from IPython import embed

from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import MultiTaskDatasetGRAM
from processor import (
    DistMultiDataTaskSampler,
    SingleMultiDataTaskSampler,
    CollatorGen,
    Collator,
    CollatorGRAM,
)


def get_dataset_gram(args, model_gen, tokenizer, phase=0, regenerate=False):
    datasets = args.datasets.split(",")
    train_all_datasets_id = []
    train_all_datasets_rec = []
    valid_all_datasets = []
    for data in datasets:
        TrainDatasetRec = MultiTaskDatasetGRAM(
            args, data, "train", model_gen, tokenizer, phase, regenerate=regenerate
        )
        TrainDatasetID = MultiTaskDatasetGRAM(
            args, data, "train", model_gen, tokenizer, phase, regenerate=regenerate
        )

        train_all_datasets_id.append(TrainDatasetID)
        train_all_datasets_rec.append(TrainDatasetRec)

    TrainSetID = ConcatDataset(
        train_all_datasets_id
    )  # Not used (Just for compatibility)
    TrainSetRec = ConcatDataset(train_all_datasets_rec)
    ValidSet = None

    return TrainSetID, TrainSetRec, ValidSet


def get_loader_gram(
    args, tokenizer, TrainSetID, TrainSetRec, ValidSet, rank=0, model_config=None
):
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node == 1:
        args.distributed = 0

    if args.dist_sampler == 0:
        train_sampler_id = (
            DistMultiDataTaskSampler(
                TrainSetID,
                args.id_batch_size,
                args.world_size,
                rank,
                args.seed,
                shuffle=True,
            )
            if args.distributed
            else SingleMultiDataTaskSampler(
                TrainSetID, args.id_batch_size, args.seed, shuffle=True
            )
        )
        train_sampler_rec = (
            DistMultiDataTaskSampler(
                TrainSetRec,
                args.rec_batch_size,
                args.world_size,
                rank,
                args.seed,
                shuffle=True,
            )
            if args.distributed
            else SingleMultiDataTaskSampler(
                TrainSetRec, args.rec_batch_size, args.seed, shuffle=True
            )
        )
    else:
        train_sampler_id = DistributedSampler(TrainSetID) if args.distributed else None
        train_sampler_rec = (
            DistributedSampler(TrainSetRec) if args.distributed else None
        )
    collator_gen = CollatorGen(tokenizer=tokenizer)
    collator_rec = CollatorGRAM(tokenizer=tokenizer, args=args, mode="train")
    train_loader_id = DataLoader(
        dataset=TrainSetID,
        sampler=train_sampler_id,
        batch_size=args.id_batch_size,
        collate_fn=collator_gen,
        shuffle=False,
    )
    train_loader_rec = DataLoader(
        dataset=TrainSetRec,
        sampler=train_sampler_rec,
        batch_size=args.rec_batch_size,
        collate_fn=collator_rec,
        shuffle=False,
    )
    valid_loader = None

    return train_loader_id, train_loader_rec, valid_loader


# temporal for args.debug_test_on_train - distributed not supported
def get_loader_gram_train(args, tokenizer, TrainSetRec):
    train_sampler_rec = SingleMultiDataTaskSampler(
        TrainSetRec, 1, args.seed, shuffle=True
    )
    collator_rec = CollatorGRAM(tokenizer=tokenizer, args=args, mode="test")
    train_loader_rec = DataLoader(
        dataset=TrainSetRec,
        sampler=train_sampler_rec,
        batch_size=1,
        collate_fn=collator_rec,
        shuffle=False,
    )
    return train_loader_rec


def get_loader(args, tokenizer, TrainSetID, TrainSetRec, ValidSet, rank=0):
    # generate training validation loader.
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node == 1:
        args.distributed = 0

    if args.dist_sampler == 0:
        train_sampler_id = (
            DistMultiDataTaskSampler(
                TrainSetID,
                args.id_batch_size,
                args.world_size,
                rank,
                args.seed,
                shuffle=True,
            )
            if args.distributed
            else SingleMultiDataTaskSampler(
                TrainSetID, args.id_batch_size, args.seed, shuffle=True
            )
        )
        train_sampler_rec = (
            DistMultiDataTaskSampler(
                TrainSetRec,
                args.rec_batch_size,
                args.world_size,
                rank,
                args.seed,
                shuffle=True,
            )
            if args.distributed
            else SingleMultiDataTaskSampler(
                TrainSetRec, args.rec_batch_size, args.seed, shuffle=True
            )
        )
    else:
        train_sampler_id = DistributedSampler(TrainSetID) if args.distributed else None
        train_sampler_rec = (
            DistributedSampler(TrainSetRec) if args.distributed else None
        )

    collator_gen = CollatorGen(tokenizer)
    collator_rec = Collator(tokenizer)
    train_loader_id = DataLoader(
        dataset=TrainSetID,
        sampler=train_sampler_id,
        batch_size=args.id_batch_size,
        collate_fn=collator_gen,
        shuffle=False,
    )
    train_loader_rec = DataLoader(
        dataset=TrainSetRec,
        sampler=train_sampler_rec,
        batch_size=args.rec_batch_size,
        collate_fn=collator_rec,
        shuffle=False,
    )
    valid_loader = None

    return train_loader_id, train_loader_rec, valid_loader


# for adding ml ds

def detect_dataset_family(dataset_name):
    """
    自动检测数据集类型并返回相应配置
    """
    amazon_datasets = ['Beauty', 'Toys', 'Sports', 'Books', 'Electronics']
    movielens_datasets = ['ML32M', 'ML1M', 'ML10M', 'ML100K', 'MovieLens']
    
    if dataset_name in amazon_datasets:
        return 'Amazon'
    elif dataset_name in movielens_datasets or 'ML' in dataset_name:
        return 'MovieLens'
    elif dataset_name in ['Yelp']:
        return 'Other'
    else:
        return 'Unknown'

def get_dataset_config(dataset_name):
    """
    根据数据集返回特定配置
    """
    family = detect_dataset_family(dataset_name)
    
    config = {
        'Amazon': {
            'simplified_metadata': False,
            'item_prompt_max_len': 128,
            'max_his': 20,
            'disable_fine_grained_fusion': False,
            'item_prompt_type': 'all_text',
            'rec_epochs': 30,
            'hierarchical_clusters': 128,
            'num_cf': 10
        },
        'MovieLens': {
            'simplified_metadata': True,
            'item_prompt_max_len': 64,
            'max_his': 30,
            'disable_fine_grained_fusion': True,
            'item_prompt_type': 'simplified_text',
            'rec_epochs': 20,
            'hierarchical_clusters': 128,
            'num_cf': 5
        },
        'Other': {
            'simplified_metadata': False,
            'item_prompt_max_len': 128,
            'max_his': 20,
            'disable_fine_grained_fusion': False,
            'item_prompt_type': 'all_text',
            'rec_epochs': 30,
            'hierarchical_clusters': 128,
            'num_cf': 10
        }
    }
    
    return config.get(family, config['Other'])

def apply_dataset_specific_args(args):
    """
    根据数据集自动调整参数
    """
    if args.dataset_family == "auto":
        config = get_dataset_config(args.datasets)
        
        # 只在用户没有手动设置时应用自动配置
        for key, value in config.items():
            if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
        
        family = detect_dataset_family(args.datasets)
        print(f"🎯 Auto-detected dataset family for {args.datasets}: {family}")
        print(f"📋 Applied configuration: {config}")
    
    return args