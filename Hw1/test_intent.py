import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import torch
from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

import csv

def test_label_csv(ids, pred_labels, pred_file):
    header = ["id", "intent"]
    with open(pred_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
        for id, pred_label in zip(ids, pred_labels):
            writer.writerow([id, pred_label])
    
    # print("test csv finished")

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    
    # TODO: crecate DataLoader for test dataset
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
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

    # TODO: predict dataset
    ids = [d["id"] for d in data]

    # predict y (idx)
    y_pred = torch.tensor([], dtype=torch.int8).to(args.device)
    for _, d in enumerate(test_loader):
        x_test = d["tokens"].to(args.device)
        outputs = model(x_test)
        
        yi_pred = torch.argmax(outputs, axis=1).to(args.device)
        y_pred = torch.cat((y_pred, yi_pred), 0).to(args.device)
    
    # predict real
    labels_pred = []
    for yi_pred in y_pred:
        label_pred = dataset.idx2label(yi_pred.item())
        labels_pred.append(label_pred)

    # TODO: write prediction to file (args.pred_file)
    test_label_csv(ids, labels_pred, args.pred_file)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/intent/best.pt",
    )
    parser.add_argument("--pred_file", type=Path, default="./pred/intent/pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.pred_file.parent.mkdir(parents=True, exist_ok=True)
    main(args)
