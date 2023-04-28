# Unseen user - Course prediction

## 1. Environment (Anaconda)
```shell
conda create -n ADL_FP python=3.9
conda activate ADL_FP
pip install -r requirements.in
```
## 2. Training

```shell
python3.9 best_seller.py --do_train \
--users_file ./data/users.csv --train_file ./data/train.csv
```
## 2. Validation

```shell
python3.9 best_seller.py --do_val \
--users_file ./data/users.csv --val_file ./data/val_unseen.csv --val_pred_file ./pred_val_unseen.csv
```
```shell
python3.9 metric.py --reference ./data/val_unseen.csv --submission ./pred_val_unseen.csv
```

## 3. Predicting Script
```shell
bash ./download.sh
bash ./run.sh /path/to/users.csv /path/to/test.csv /path/to/output.csv
```