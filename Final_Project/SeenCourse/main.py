import json
import torch
import torch.nn as nn
from tqdm import tqdm

from utils import parse_args, save_ckpt, export_pred, set_seeds
from preprocess import load_dataset
from model import HahowRanker
from eval import mapk


def main(args):
    set_seeds()

    train_dataloader, eval_dataloader, test_dataloader, lookup = load_dataset(args)
    records = {'args': {k: str(v) for k, v in vars(args).items()}, 'train_loss': [], 'eval_loss': [], 'eval_mapk': []}

    model = HahowRanker(
        num_courses=len(lookup['course_id2idx']),
        ignore_idx=train_dataloader.dataset.ignore_index
    ).to(args.device)
    if args.pretrained_file:
        # with open(args.output_dir / args.record_file, 'r') as f:
        #     records = json.load(f)
        model.load_state_dict(torch.load(args.pretrained_file))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataloader.dataset.ignore_index)

    if args.do_train:
        try:
            for epoch in range(args.num_epoch):
                print(f"\nEpoch: #{epoch}")
                train(args, model, train_dataloader, optimizer, loss_fn, records)
                evaluate(args, model, eval_dataloader, records)
                save_ckpt(args, model, records)
        except KeyboardInterrupt as e:
            pass

    evaluate(args, model, eval_dataloader)
    if args.do_test:
        evaluate(args, model, test_dataloader, is_test=True)


def train(args, model, dataloader, optimizer, loss_fn, records=[]):
    model.train()
    sum_loss = 0
    for batch_data in tqdm(dataloader, desc='train'):
        batch_data = [batch_tensor.to(args.device) for batch_tensor in batch_data]
        batch_tokens, batch_labels, batch_padding_masks, batch_attrs = batch_data

        logits = model(batch_tokens, batch_padding_masks, batch_attrs)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch_labels.reshape(-1))
        sum_loss += loss.item() if not torch.isnan(loss) else 0

        # ! Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"\tTrain loss: {sum_loss/len(dataloader)}")
    records['train_loss'].append(sum_loss/len(dataloader))


def evaluate(args, model, dataloader, records={}, is_test=False):
    model.eval()
    preds = []
    row_idx = 0
    lookup = dataloader.dataset.lookup
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc='eval' if not is_test else 'test'):
            batch_data = [batch_tensor.to(args.device) for batch_tensor in batch_data]
            batch_tokens, batch_padding_masks, batch_attrs = batch_data

            logits = model(batch_tokens, batch_padding_masks, batch_attrs)
            mask_logit = logits[:, -1, :]

            for course_idxs, done_course_idxs in zip(mask_logit.argsort(dim=1, descending=True), batch_tokens):
                pred = []
                user_idx = dataloader.dataset.data[row_idx]['user_idx']
                done_course_idxs = {x.item() for x in done_course_idxs}
                if is_test:
                    done_course_idxs.update(lookup['user_idx2done_courses'][user_idx])
                for course_idx in course_idxs:
                    if len(pred) >= 10:
                        break
                    if course_idx.item() not in done_course_idxs:
                        pred.append(course_idx.item())
                preds.append(pred)
                row_idx += 1

    for i in range(len(preds)):
        preds[i] = [lookup['course_idx2id'][pred_id] for pred_id in preds[i] if pred_id in lookup['course_idx2id']]

    if is_test:
        export_pred(preds, dataloader.dataset.dataset, args.output_dir / args.output_file)
    else:
        actuals = []
        for entry in dataloader.dataset.data:
            actuals.append([lookup['course_idx2id'][course_idx] for course_idx in entry['course_idxs']])
        score = mapk(actuals, preds)
        print(f"\tEval mapk: {score}")
        if 'eval_mapk' in records:
            records['eval_mapk'].append(score)


if __name__ == "__main__":
    args = parse_args()
    main(args)
