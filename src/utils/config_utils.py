"""
配置工具，用于处理数据集特定的配置
"""

def update_config_for_dataset(config, args):
    """
    根据args更新config，确保模型能接收到简化处理标志
    """
    if hasattr(args, 'simplified_metadata'):
        config.simplified_metadata = args.simplified_metadata
    
    if hasattr(args, 'disable_fine_grained_fusion'):
        config.disable_fine_grained_fusion = args.disable_fine_grained_fusion
    
    if hasattr(args, 'dataset_family'):
        config.dataset_family = args.dataset_family
    
    return config