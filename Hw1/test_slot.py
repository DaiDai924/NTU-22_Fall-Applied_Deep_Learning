import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import torch
from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

import csv

def test_label_csv(ids, pred_labels, pred_file):
    header = ["id", "tags"]
    with open(pred_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
        for id, pred_label in zip(ids, pred_labels):
            str_pred_label = ' '.join(pred_label)
            writer.writerow([id, str_pred_label])
    
    # print("test csv finished")

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
    
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    # load weights into model
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt["model_state_dict"])

    ids = [d["id"] for d in data]
    len_tokens = [len(d["tokens"]) for d in data]

    # predict y (idx), and convert to target y (label)
    labels_pred = []
    idx = 0
    for _, d in enumerate(test_loader):
        x_test = d["tokens"].to(args.device)
        outputs = model(x_test)
        
        for _, outputs_i in enumerate(outputs):
            outputs_i = outputs_i[:len_tokens[idx]] # only consider the real length of the tokens
            yi_pred = torch.argmax(outputs_i, axis=1).to(args.device) # [len_token]
            label_pred = [dataset.idx2label(tag_pred.item()) for tag_pred in yi_pred]
            labels_pred.append(label_pred)
            idx += 1

    test_label_csv(ids, labels_pred, args.pred_file)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/slot/best.pt",
    )
    parser.add_argument("--pred_file", type=Path, default="./pred/slot/pred.tags.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.pred_file.parent.mkdir(parents=True, exist_ok=True)
    main(args)
