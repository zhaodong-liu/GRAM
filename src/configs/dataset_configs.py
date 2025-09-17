"""
数据集特定配置
"""

DATASET_CONFIGS = {
    'Amazon': {
        'Beauty': {
            'simplified_metadata': False,
            'item_prompt_max_len': 128,
            'max_his': 20,
            'hierarchical_clusters': 128,
            'num_cf': 10,
            'use_multi_granular_fusion': True,
            'rec_epochs': 30
        },
        'Toys': {
            'simplified_metadata': False,
            'item_prompt_max_len': 128,
            'max_his': 20,
            'hierarchical_clusters': 32,
            'num_cf': 5,
            'use_multi_granular_fusion': True,
            'rec_epochs': 30
        },
        'Sports': {
            'simplified_metadata': False,
            'item_prompt_max_len': 128,
            'max_his': 20,
            'hierarchical_clusters': 32,
            'num_cf': 10,
            'use_multi_granular_fusion': True,
            'rec_epochs': 30
        }
    },
    'MovieLens': {
        'ML32M': {
            'simplified_metadata': True,
            'item_prompt_max_len': 64,
            'max_his': 30,
            'hierarchical_clusters': 128,
            'num_cf': 5,
            'use_multi_granular_fusion': False,
            'rec_epochs': 20
        },
        'ML1M': {
            'simplified_metadata': True,
            'item_prompt_max_len': 64,
            'max_his': 25,
            'hierarchical_clusters': 64,
            'num_cf': 5,
            'use_multi_granular_fusion': False,
            'rec_epochs': 25
        }
    },
    'Other': {
        'Yelp': {
            'simplified_metadata': False,
            'item_prompt_max_len': 128,
            'max_his': 20,
            'hierarchical_clusters': 32,
            'num_cf': 5,
            'use_multi_granular_fusion': True,
            'rec_epochs': 30
        }
    }
}

def get_config(dataset_family, dataset_name):
    """
    获取特定数据集的配置
    """
    return DATASET_CONFIGS.get(dataset_family, {}).get(dataset_name, {})