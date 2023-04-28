import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def validate(model, dataloader):
    num_data = 0
    num_success = 0
    
    y_true = []
    y_pred = []
    for _, batch in enumerate(dataloader):
        x = batch["tokens"].to(args.device)
        y = batch["tags"].to(args.device)
        len_tokens = batch["length"].to(args.device)
        num_data += len(y)

        outputs = model(x)   # [max_len, num_class]
        
        for i, outputs_i in enumerate(outputs):
            outputs_i = outputs_i[:len_tokens[i]]
            
            yi_pred = torch.argmax(outputs_i, axis=1).to(args.device)   # [real_token_len]
            y_pred.append(yi_pred.tolist())
            
            yi = y[i][:len_tokens[i]]
            y_true.append(yi.tolist())
            
            pred_success = True
            for idx_tag in range(len(yi)):
                if yi[idx_tag] != yi_pred[idx_tag]:
                    pred_success = False
                    break
            if pred_success:
                num_success += 1
    
    # index to label
    for i, yi in enumerate(y_true):
        y_true[i] = [dataloader.dataset.idx2label(idx) for idx in yi]
    for i, yi_pred in enumerate(y_pred):
        y_pred[i] = [dataloader.dataset.idx2label(idx) for idx in yi_pred]  
    
    joint_val_acc = num_success / num_data
    token_val_acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, mode='strict', scheme=IOB2)
    
    print("Joint accuracy:", joint_val_acc)
    print("Token accuracy:", token_val_acc)
    print("Classification report:\n", report)
    
    return joint_val_acc, token_val_acc, report

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

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    # TODO: crecate DataLoader for train / dev datasets
    train_loader = torch.utils.data.DataLoader(dataset=datasets[TRAIN], batch_size=args.batch_size, shuffle=True, collate_fn=datasets[TRAIN].collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset=datasets[DEV], batch_size=args.batch_size, shuffle=True, collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqTagger(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional, num_class=datasets[TRAIN].num_classes)
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
            y_train = batch["tags"].to(args.device)

            optimizer.zero_grad()    # clear gradients

            outputs = model(x_train) # outputs: [batch_size, max_len, num_class]

            # resize for Cross Entropy Loss
            #   input: (N, C) -> resize_outputs(batch_size * max_len, num_class)
            #   target: (N) -> resize_train_y(batch_size * max_len)
            resize_outputs = outputs.view(outputs.shape[0] * outputs.shape[1], -1)
            resize_train_y = y_train.view(y_train.shape[0] * y_train.shape[1])
                
            loss = loss_fn(resize_outputs, resize_train_y)
            
            loss.backward()          # calculate new gradients
            optimizer.step()         # update network parameters

        # TODO: Evaluation loop - calculate accuracy and save model weights
        print("Epoch:", epoch + 1)
        joint_val_acc, token_val_acc, report = validate(model, val_loader)
        
        if joint_val_acc > best_accuracy:
            best_accuracy = joint_val_acc
            save_ckpt(args.ckpt_dir, model, epoch, optimizer, loss, best=True)
        else:
            save_ckpt(args.ckpt_dir, model, epoch, optimizer, loss)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
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
    parser.add_argument("--batch_size", type=int, default=16)

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
