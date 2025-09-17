def load_movielens_data(data_path, dataset_name):
    """
    加载MovieLens数据
    """
    dataset_dir = os.path.join(data_path, dataset_name)
    
    # 加载用户序列
    sequences_file = os.path.join(dataset_dir, f"{dataset_name}_sequences.json")
    with open(sequences_file, 'r') as f:
        user_sequences = json.load(f)
    
    # 转换格式
    user_sequence_dict = {}
    for user_id, sequence in user_sequences.items():
        user_sequence_dict[int(user_id)] = [int(item_id) for item_id in sequence]
    
    # 加载电影元数据
    metadata_file = os.path.join(dataset_dir, f"{dataset_name}_item_metadata.json")
    with open(metadata_file, 'r') as f:
        item_metadata = json.load(f)
    
    # 转换格式
    item_meta_dict = {}
    for item_id, metadata in item_metadata.items():
        item_meta_dict[int(item_id)] = {
            'title': metadata['title'],
            'genres': metadata.get('genres', []),
            'year': metadata.get('year', ''),
            'text': metadata.get('text', '')
        }
    
    return user_sequence_dict, item_meta_dict