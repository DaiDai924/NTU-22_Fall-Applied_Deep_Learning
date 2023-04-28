from typing import List, Dict

from torch.utils.data import Dataset
import torch
from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        """
        Input:
            samples: List[Dict], a set of samples in senteces with intent
        Output:
            batch: Dict, {"tokens": tokenized word index with padding(2d tensor)}, {"intent": intent index(tensor)}
        """

        batch = {}
        batch["tokens"] = [sample["text"].split() for sample in samples]
        batch["tokens"] = torch.tensor(self.vocab.encode_batch(batch["tokens"], self.max_len))
        # "intent" only exist in train/validation samples, not in test samples
        if "intent" in samples[0]:
            batch["intent"] = torch.tensor([self.label2idx(sample["intent"]) for sample in samples])

        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: tag for tag, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.ignore_idx = -100

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        """
        Input:
            samples: List[Dict], a set of samples in senteces with tags
        Output:
            batch: Dict, {"tokens": tokenized word index with padding(2d tensor)}, {"tags": tag index(tensor)}
        """
        
        batch = {}
        batch["length"] = torch.tensor([len(sample["tokens"]) for sample in samples])
        batch["tokens"] = [sample["tokens"] for sample in samples]
        batch["tokens"] = torch.tensor(self.vocab.encode_batch(batch["tokens"], self.max_len))
        # "tags" only exist in train/validation samples, not in test samples
        if "tags" in samples[0]:
            batch["tags"] = [[self.label2idx(tag) for tag in sample["tags"]] for sample in samples]
            batch["tags"] = torch.tensor(pad_to_len(batch["tags"], self.max_len, self.ignore_idx))
        
        return batch