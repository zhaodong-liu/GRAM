import copy
import torch
from torch.utils.data import Dataset
from utils import indexing
import torch.distributed as dist
import logging

from time import time


class MultiTaskDatasetGRAM(Dataset):
    def __init__(
        self, args, dataset, mode, model_gen, tokenizer, phase=0, regenerate=False
    ):
        super().__init__()
        assert (
            regenerate == False
        ), "regenerate is not supported in MultiTaskDatasetGRAM. id is fixed"
        assert args.sample_num

        self.model_gen = model_gen
        self.tokenizer = tokenizer
        self.data_path = args.data_path
        self.dataset = dataset
        self.tasks = args.tasks.split(",")
        if args.sample_prompt > 0:
            assert len(self.tasks) == len(
                args.sample_num.split(",")
            ), "prompt sample number does not match task number"
        self.mode = mode
        self.args = args
        self.phase = phase

        from utils.dataset_utils import detect_dataset_family
    
        # 新增：检测是否使用简化处理 (for ml ds
        self.use_simplified_processing = getattr(args, 'simplified_metadata', False)
        self.dataset_family = detect_dataset_family(dataset)
        self.disable_fine_grained = getattr(args, 'disable_fine_grained_fusion', False)
        
        if self.rank == 0:
            print(f"📊 Dataset {dataset} family: {self.dataset_family}")
            print(f"🔧 Using simplified processing: {self.use_simplified_processing}")
            print(f"🔧 Disable fine-grained fusion: {self.disable_fine_grained}")

        self.rank = args.rank
        self.skip_empty_his = args.skip_empty_his
        self.reverse_history = args.reverse_history
        self.user_id_without_target_item = args.user_id_without_target_item
        self.id_linking = args.id_linking

        if self.rank == 0:
            logging.info(f"Generating data for {self.dataset} dataset")

        self.max_his = args.max_his
        self.his_sep = args.his_sep

        # apply indexing method and avoid generate data multiple times
        if args.distributed:
            if self.rank == 0:
                logging.info("Reindex data with generative indexing method")
                indexing.gram_indexing(
                    data_path=self.data_path,
                    dataset=self.dataset,
                    model_gen=self.model_gen,
                    tokenizer=self.tokenizer,
                    regenerate=regenerate,
                    phase=self.phase,
                    args=self.args,
                    user_id_without_target_item=self.user_id_without_target_item,
                    id_linking=self.id_linking,
                )
                dist.barrier()
            else:
                dist.barrier()
            self.user_seq_dict, self.item2input, self.item2lexid = (
                indexing.gram_indexing(
                    data_path=self.data_path,
                    dataset=self.dataset,
                    model_gen=self.model_gen,
                    tokenizer=self.tokenizer,
                    regenerate=regenerate,
                    phase=self.phase,
                    args=self.args,
                    user_id_without_target_item=self.user_id_without_target_item,
                    id_linking=self.id_linking,
                )
            )
        else:
            logging.info("Reindex data with generative indexing method")
            self.user_seq_dict, self.item2input, self.item2lexid = (
                indexing.gram_indexing(
                    data_path=self.data_path,
                    dataset=self.dataset,
                    model_gen=self.model_gen,
                    tokenizer=self.tokenizer,
                    regenerate=regenerate,
                    phase=self.phase,
                    args=self.args,
                    user_id_without_target_item=self.user_id_without_target_item,
                    id_linking=self.id_linking,
                )
            )
        self.all_items = list(self.item2lexid.values())

        # load data
        if self.mode == "train":
            if self.rank == 0:
                logging.info("loading training data")
            self.data_samples = self.load_train()
        elif self.mode == "validation":
            self.data_samples = self.load_validation()
            if self.rank == 0:
                logging.info("loading validation data")
        else:
            raise NotImplementedError

        if self.args.debug_train_100:
            self.data_samples = self.data_samples[:100]
            if self.rank == 0:
                logging.info(
                    ">>>> Debug mode: only use 100 samples for training (MultiTaskDatasetGRAM)"
                )

        self.get_prompt_info()
        self.construct_sentence()

    def get_positive(self):
        """
        Get a dict of set to save the positive interactions for negative candidate sampling
        """
        positive = dict()
        for user in self.user_seq_dict:
            if self.mode == "train":
                positive[user] = set(self.user_seq_dict[user][:-2])
            if self.mode == "validation":
                positive[user] = set(self.user_seq_dict[user][:-1])
            if self.mode == "test":
                positive[user] = set(self.user_seq_dict[user])
        return positive

    def shuffle(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)

        for task in self.task_data:
            indices = torch.randperm(len(self.task_data[task]), generator=g).tolist()
            self.task_data[task] = [self.task_data[task][i] for i in indices]

    def get_prompt_info(self):
        """
        Simplified version for single task with hardcoded prompts
        """
        # For single task, just create simple task_data mapping
        self.task_data = {self.tasks[0]: list(range(len(self.data_samples)))}
        # These might not be needed for single task, but keeping for compatibility
        self.task_prompt_num = [1]
        self.task_index = [len(self.data_samples)]

    def load_train(self):
        """
        Load training data samples

        already adopting data augmentation (for loop - range(len(items)))
        - Original: A B C D -> E
        - Augmented: A -> B, A B -> C, A B C -> D, A B C D -> E
        """
        st = time()
        data_samples = []
        for user in self.user_seq_dict:
            items = self.user_seq_dict[user][:-2]
            for i in range(len(items)):
                if i == 0:
                    if self.skip_empty_his > 0:
                        continue
                one_sample = dict()
                one_sample["dataset"] = self.dataset
                one_sample["user_id"] = user  # 'A1YJEY40YUW4SE'
                one_sample["target"] = items[i]  # 'B004ZT0SSG'
                one_sample["target_lex_id"] = self.item2lexid[
                    items[i]
                ]  # 'red shatter crackle nail polish e55'

                history = items[:i]
                if self.max_his > 0:
                    history = history[-self.max_his :]
                one_sample["history"] = self.his_sep.join(
                    history
                )  # 'B00KAL5JAU ; B00KHGIK54'
                one_sample["history_input"] = [
                    self.item2input[h] for h in history
                ]  # ['nail lacquer simmer and shimmer', ...]

                if self.reverse_history:
                    tmp_history = copy.deepcopy(one_sample["history"]).split(
                        self.his_sep
                    )
                    one_sample["history"] = self.his_sep.join(tmp_history[::-1])
                    one_sample["history_input"] = one_sample["history_input"][::-1]

                # TODO prefix hardcoded
                history_lex_ids = [
                    self.item2lexid[h] for h in history[::-1]
                ]  # reverse order
                one_sample["history_lex_id"] = self.his_sep.join(
                    history_lex_ids
                )  # 'nail lacquer simmer and shimmer ; red shatter crackle nail polish e55 ; ...'

                data_samples.append(one_sample)

        if self.rank == 0 and self.args.verbose_input_output:
            logging.info(f">> Load training data time: {time()-st} s")

        return data_samples

    def load_validation(self):
        """
        Load validation data samples
        """
        st = time()
        data_samples = []
        for user in self.user_seq_dict:
            items = self.user_seq_dict[user]
            one_sample = dict()
            one_sample["dataset"] = self.dataset
            one_sample["user_id"] = user
            one_sample["target"] = items[-2]
            one_sample["target_lex_id"] = self.item2lexid[items[-2]]

            history = items[:-2]
            if self.max_his > 0:
                history = history[-self.max_his :]
            one_sample["history"] = self.his_sep.join(history)
            one_sample["history_input"] = [self.item2input[h] for h in history]

            if self.reverse_history:
                tmp_history = copy.deepcopy(one_sample["history"]).split(self.his_sep)
                one_sample["history"] = self.his_sep.join(tmp_history[::-1])
                one_sample["history_input"] = one_sample["history_input"][::-1]

            history_lex_ids = [self.item2lexid[h] for h in history[::-1]]
            one_sample["history_lex_id"] = self.his_sep.join(
                history_lex_ids
            )  # 'nail lacquer simmer and shimmer ; red shatter crackle nail polish e55 ; ...'

            data_samples.append(one_sample)

        if self.rank == 0 and self.args.verbose_input_output:
            logging.info(f">> Load validation data time: {time()-st} s")

        return data_samples

    def construct_sentence(self):
        """
        Make data_samples into model input, output pairs
        data_samples: {
            'dataset': 'Beauty',
            'user_id': 'A2CG5Y82ZZNY6W',
            'target': 'B00KHH2VOY',
            'target_lex_id': 'dead sea salt soap body soap dead sea',
            'history': 'B00KAL5JAU ; B00KHGIK54',
            'history_input': ['dead sea salt deep hair conditioner natural curly',
                                'adovia facial serum anti aging'],
            'history_lex_id': 'dead sea salt deep hair conditioner natural curly ; adovia facial serum anti aging',
            }
        """

        st = time()
        self.data = {}
        self.data["input"] = []
        self.data["output"] = []
        self.data["user_id"] = []
        for i in range(len(self.data_samples)):
            datapoint = self.data_samples[i]
            input_sample = []

            sequence_text = datapoint["history_lex_id"]
            user_sentence = f"What would user purchase after {sequence_text} ?"
            input_sample.append(user_sentence)

            history_input = datapoint["history_input"]
            input_sample += history_input

            self.data["input"].append(input_sample)
            self.data["output"].append(datapoint["target_lex_id"])
            self.data["user_id"].append(datapoint["user_id"])

        if self.rank == 0 and self.args.verbose_input_output:
            logging.info(f">> Constructing sentence time: {time()-st} s")
            logging.info(
                f">> Input: {self.data['input'][-1]} \n>> Output: {self.data['output'][-1]}"
            )

    def __len__(self):
        return len(self.data["input"])

    def __getitem__(self, idx):
        return {
            "input": self.data["input"][idx],
            "output": self.data["output"][idx],
            "user_id": self.data["user_id"][idx],
        }

    
    def construct_item_prompt_adaptive(self, item_metadata):
        """
        根据数据集类型构建item prompt
        """
        if self.use_simplified_processing or self.dataset_family == 'MovieLens':
            return self._construct_simplified_movie_prompt(item_metadata)
        else:
            return self._construct_complex_amazon_prompt(item_metadata)
    
    def _construct_simplified_movie_prompt(self, item_metadata):
        """
        MovieLens简化prompt构建
        """
        title = item_metadata.get('title', '')
        genres = item_metadata.get('genres', [])
        year = item_metadata.get('year', '')
        
        # 构建简化的电影prompt
        if isinstance(genres, list):
            genres_str = ', '.join(genres)
        else:
            genres_str = str(genres)
            
        prompt_parts = [title]
        if genres_str and genres_str != 'nan' and genres_str.strip():
            prompt_parts.append(f"Genres: {genres_str}")
        if year and str(year) != 'nan' and str(year).strip():
            prompt_parts.append(f"Year: {year}")
        
        return " | ".join(prompt_parts)
    
    def _construct_complex_amazon_prompt(self, item_metadata):
        """
        Amazon复杂prompt构建（保持原版逻辑）
        """
        # 这里保持你现有的复杂prompt构建逻辑不变
        # 如果没有现有逻辑，可以使用以下简化版本：
        
        title = item_metadata.get('title', '')
        brand = item_metadata.get('brand', '')
        categories = item_metadata.get('categories', [])
        description = item_metadata.get('description', '')
        
        prompt_parts = []
        if title:
            prompt_parts.append(f"title: {title}")
        if brand:
            prompt_parts.append(f"brand: {brand}")
        if categories:
            if isinstance(categories, list):
                categories_str = ', '.join(categories)
            else:
                categories_str = str(categories)
            prompt_parts.append(f"categories: {categories_str}")
        if description:
            prompt_parts.append(f"description: {description}")
        
        return " | ".join(prompt_parts)
    
    def get_multi_granular_prompts_adaptive(self, user_sequence, item_metadata_dict):
        """
        条件性multi-granular prompt构建
        """
        if self.use_simplified_processing or self.disable_fine_grained:
            # MovieLens: 简化处理，不分离user/item prompts
            return self._get_simplified_prompts(user_sequence, item_metadata_dict)
        else:
            # Amazon: 原版multi-granular处理
            return self._get_complex_prompts(user_sequence, item_metadata_dict)
    
    def _get_simplified_prompts(self, user_sequence, item_metadata_dict):
        """
        简化版本：直接合并用户历史和item信息
        """
        combined_prompts = []
        
        for item_id in user_sequence[-self.args.max_his:]:  # 取最近的历史
            if item_id in item_metadata_dict:
                item_prompt = self.construct_item_prompt_adaptive(item_metadata_dict[item_id])
                combined_prompts.append(item_prompt)
        
        # 构建单一的用户prompt
        if combined_prompts:
            user_prompt = f"User watched: {' -> '.join(combined_prompts[-10:])}"  # 最近10个
        else:
            user_prompt = "User watched: [empty history]"
        
        return {
            'user_prompt': user_prompt,
            'item_prompts': None,  # 不使用分离的item prompts
            'fusion_type': 'simplified'
        }
    
    def _get_complex_prompts(self, user_sequence, item_metadata_dict):
        """
        复杂版本：保持原版multi-granular分离处理
        """
        # 这里保持你现有的复杂处理逻辑
        # 如果没有现有逻辑，返回简化版本
        return self._get_simplified_prompts(user_sequence, item_metadata_dict)