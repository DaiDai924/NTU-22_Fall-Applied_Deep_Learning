import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab

from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def validate(model, dataloader):
    num_data = 0
    num_success = 0

    for _, batch in enumerate(dataloader):
        x = batch["tokens"].to(args.device)
        y = batch["intent"].to(args.device)
        num_data += len(y)

        outputs = model(x)
        y_pred = torch.argmax(outputs, axis=1).to(args.device)

        num_success += sum(i == j for i, j in zip(y_pred, y)).item()

    return num_success / num_data

def save_ckpt(ckpt_dir, model, epoch, optimizer, loss, best=False):
    model_ckpt_path = ckpt_dir / "model_{}.pt".format(epoch + 1)
    
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, model_ckpt_path)
    print("Save checkpoint of model_{}. ".format(epoch + 1))
    
    if best:
        best_ckpt_path = ckpt_dir / "best.pt"
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                    }, best_ckpt_path)
        print("model_{} is the best model now. ".format(epoch + 1))

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    # TODO: crecate DataLoader for train / dev datasets
    train_loader = torch.utils.data.DataLoader(dataset=datasets[TRAIN], batch_size=args.batch_size, shuffle=True, collate_fn=datasets[TRAIN].collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset=datasets[DEV], batch_size=args.batch_size, shuffle=True, collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional, num_class=datasets[TRAIN].num_classes)
    model = model.to(args.device)

    # TODO: init optimizer and loss function
    optimizer = torch.optim.Adam(params = model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_accuracy = 0.0

    model.train()
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:

        # TODO: Training loop - iterate over train dataloader and update model weights
        for _, batch in enumerate(train_loader):
            x_train = batch["tokens"].to(args.device)
            y_train = batch["intent"].to(args.device)

            optimizer.zero_grad()    # clear gradients

            outputs = model(x_train) # outputs: [batch_size, num_class]
            loss = loss_fn(outputs, y_train)
            
            loss.backward()          # calculate new gradients
            optimizer.step()         # update network parameters

        # TODO: Evaluation loop - calculate accuracy and save model weights
        val_accuracy = validate(model, val_loader)
        print("Validation accuracy of epoch {}:".format(epoch + 1), val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_ckpt(args.ckpt_dir, model, epoch, optimizer, loss, best=True)
        else:
            save_ckpt(args.ckpt_dir, model, epoch, optimizer, loss)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=20)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
