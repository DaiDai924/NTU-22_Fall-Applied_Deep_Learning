# Homework 1 ADL NTU

## Environment
```shell
# use conda
make
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection
### Train
```shell
python train_intent.py --data_dir <data_dir> --cache_dir <cache_dir> --ckpt_dir <ckpt_dir> --max_len <max_len> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <True/False> --lr <learning_rate> --batch_size <batch_size> --device <cpu, cuda, cuda:0, cuda:1> --num_epoch <num_epoch> 
```

### Test
```shell
python test_intent.py --test_file <test_file> --cache_dir <cache_dir> --ckpt_path <ckpt_path> --pred_file <pred_file> --max_len <max_len> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <True/False> --batch_size <batch_size> --device <cpu, cuda, cuda:0, cuda:1>
```

## Reproduce my result (Public: 0.89600)
```shell
bash download.sh
bash intent_cls.sh
```

## Tag detection
### Train
```shell
python train_slot.py --data_dir <data_dir> --cache_dir <cache_dir> --ckpt_dir <ckpt_dir> --max_len <max_len> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <True/False> --lr <learning_rate> --batch_size <batch_size> --device <cpu, cuda, cuda:0, cuda:1> --num_epoch <num_epoch> 
```

### Test
```shell
python test_slot.py --test_file <test_file> --cache_dir <cache_dir> --ckpt_path <ckpt_path> --pred_file <pred_file> --max_len <max_len> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <True/False> --batch_size <batch_size> --device <cpu, cuda, cuda:0, cuda:1>
```

## Reproduce my result (Public: 0.75603)
```shell
bash download.sh
bash slot_tag.sh
```
