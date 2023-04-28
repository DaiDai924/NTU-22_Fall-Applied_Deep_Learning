import os
import json
import random
import numpy as np
from pathlib import Path
from argparse import ArgumentParser, Namespace, BooleanOptionalAction
import torch
import pandas as pd


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=Path, help="Directory to the output.", default="./output/")
    parser.add_argument("--data_dir", type=Path, help="Directory to the dataset.", default="./hahow/data/")
    parser.add_argument("--cache_dir", type=Path, help="Directory to the preprocessed caches.", default="./cache/")
    parser.add_argument("--ckpt_dir", type=Path, help="Directory to save the model file.", default="./ckpt/")
    parser.add_argument("--ckpt_file", type=Path, help="Path to model checkpoint.", default='model.pt')
    parser.add_argument("--best_ckpt_file", type=Path, help="Path to best model checkpoint.", default='model_best.pt')
    parser.add_argument("--output_file", type=Path, default="pred.csv")
    parser.add_argument("--record_file", type=Path, default="records.json")

    parser.add_argument("--train_file", type=Path, default="./hahow/data/train.csv")
    parser.add_argument("--eval_file", type=Path, default="./hahow/data/val_seen.csv")
    parser.add_argument("--test_file", type=Path, default="./hahow/data/test_seen.csv")
    parser.add_argument("--courses_file", type=Path, default="./hahow/data/courses.csv")
    parser.add_argument("--pretrained_file", type=Path, default=None)

    parser.add_argument('--do_train', default=False, action=BooleanOptionalAction)
    parser.add_argument('--do_test', default=False, action=BooleanOptionalAction)

    # data
    parser.add_argument("--max_len", type=int, default=100)

    # training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=torch.device, default="cuda")

    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


best_mapk = 0


def save_ckpt(args, model, records):
    torch.save(model.state_dict(), args.ckpt_dir / args.ckpt_file)
    print(f'model saved at {args.ckpt_dir / args.ckpt_file}')
    with open(args.output_dir / args.record_file, 'w') as f:
        json.dump(records, f)
    global best_mapk
    if records['eval_mapk'][-1] > best_mapk:
        best_mapk = records['eval_mapk'][-1]
        torch.save(model.state_dict(), args.ckpt_dir / args.best_ckpt_file)
        print(f'best model saved at {args.ckpt_dir / args.best_ckpt_file}')


def export_pred(preds: torch.tensor, template: pd.DataFrame, output_path: str):
    for i in range(len(preds)):
        template.iloc[i, template.columns.get_loc('course_id')] = ' '.join(preds[i])
    template.to_csv(output_path, index=False)
    print(f'output: {output_path}')


def set_seeds():
    seed = 1234
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
