
import random
import torch
from torch.utils.data import Dataset
import pandas as pd


class HahowDataset(Dataset):
    def __init__(
        self,
        args,
        dataset: pd.DataFrame,
        lookup: dict,
        is_eval: bool = False,
        train_data: dict = None,
    ):
        self.dataset = dataset
        self.lookup = lookup
        self.is_eval = is_eval
        self.train_data = train_data
        self.max_len = args.max_len
        self.num_users = len(self.lookup['user_id2idx'])
        self.num_courses = len(self.lookup['course_id2idx'])
        self.mask_token = self.num_courses + 1
        self.ignore_index = 0

        self.data = [None] * len(dataset)
        for i, row in dataset.iterrows():
            self.data[i] = {
                'user_idx': lookup['user_id2idx'][row.user_id],
                'course_idxs': [lookup['course_id2idx'].get(course_id, 0) for course_id in row.course_id.split()]
            }
            if not is_eval:
                assert i == self.data[i]['user_idx']
                assert -1 not in self.data[i]['course_idxs']

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_eval:
            return self.get_eval_item(idx)
        else:
            return self.get_train_item(idx)

    def get_train_item(self, idx):
        tokens = self.data[idx]['course_idxs'].copy()[-self.max_len:]
        labels = [self.ignore_index] * len(tokens)
        # random.shuffle(tokens)
        mask_count = 0
        for i in range(len(tokens)):
            if random.random() < 0.3:
                labels[i] = tokens[i]
                tokens[i] = self.mask_token
                mask_count += 1
        if mask_count == 0:
            mask_idx = random.randint(0, len(tokens)-1)
            labels[mask_idx] = tokens[mask_idx]
            tokens[mask_idx] = self.mask_token

        padded_tokens, padded_labels, padding_mask = self.padding(tokens, labels)
        return torch.LongTensor(padded_tokens), torch.LongTensor(padded_labels), torch.BoolTensor(padding_mask), torch.FloatTensor(self.get_attrs(padded_tokens))

    def get_eval_item(self, idx):
        user_idx = self.data[idx]['user_idx']
        tokens = self.train_data[user_idx]['course_idxs'][-(self.max_len-1):] + [self.mask_token]
        padded_tokens, padded_labels, padding_mask = self.padding(tokens)
        return torch.LongTensor(padded_tokens), torch.BoolTensor(padding_mask), torch.FloatTensor(self.get_attrs(padded_tokens))

    def padding(self, tokens: list, labels: list = []):
        padding_len = max(0, self.max_len - len(tokens))
        padding_labels = [self.ignore_index] * padding_len
        padding_tokens = [0] * padding_len
        padding_mask = [True] * padding_len + [False] * len(tokens)
        return (padding_tokens + tokens), (padding_labels + labels), padding_mask

    def get_attrs(self, padded_tokens):
        attrs = []
        for token in padded_tokens:
            if token != 0 and token != self.mask_token:
                attrs.append(self.lookup['course_id2attr'][self.lookup['course_idx2id'][token]])
            else:
                attrs.append([0] * len(self.lookup['course_id2attr'][next(iter(self.lookup['course_id2attr']))]))
        return attrs
