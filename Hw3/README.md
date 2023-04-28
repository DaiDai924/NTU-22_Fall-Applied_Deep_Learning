# ADL22-HW3
Dataset & evaluation script for ADL 2022 homework 3

## Dataset
[download link](https://drive.google.com/file/d/186ejZVADY16RBfVjzcMcz9bal9L3inXC/view?usp=sharing)

## Environment
```shell
# use conda
make
pip install -r requirements.txt
```

## Installation
```
git clone https://github.com/moooooser999/ADL22-HW3.git
cd ADL22-HW3
pip install -e tw_rouge

# Install fixed version transformers library for fp16
# git clone https://github.com/huggingface/transformers.git
# cd ./transformers
# git checkout t5-fp16-no-nans
# pip install -e .
```

## Summarization
### Train & Evaluation
```shell
python ./train_summarization.py \
    --model_name_or_path google/mt5-small \
    --do_train \
    --do_eval \
    --train_file [path_to_train.jsonl] \
    --validation_file [path_to_validation.jsonl] \
    --source_prefix "summarize: " \
    --text_column maintext \
    --summary_column title \
    --per_device_train_batch_size [per_device_train_batch_size] \
    --per_device_eval_batch_size [per_device_eval_batch_size] \
    --optim [adamw_hf,adamw_torch,adamw_torch_xla,adamw_apex_fused,adafactor,adamw_bnb_8bit,sgd,adagrad] \
    --learning_rate [learning_rate] \
    --max_source_length [max_source_length] \
    --max_target_length [max_target_length] \
    --output_dir [output_dir] \
    --overwrite_output_dir \
    --predict_with_generate \
    --evaluation_strategy steps \
    --eval_steps [num_steps] \
    --logging_steps [num_steps]
```

### Test
```shell
python ./test.py \
    --model_name_or_path [path_to_bestmodel] \
    --test_file [path_to_test.jsonl] \
    --source_prefix "summarize: " \
    --text_column maintext \
    --output_dir [path_to_pred.jsonl] \
    --per_device_test_batch_size [per_device_test_batch_size]
    # --num_beams [num_beams] --early_stopping [True/False]
    # --top_k [top_k] --do_sample True
    # --top_q [top_q] --top_k 0 --do_sample True
    # --temperature [temperature] --do_sample True
```

### Draw learning curves 
```shell
python learning_curve.py --trainer_state_file [trainer_state_file]
```

## Reproduce my result
```shell
bash download.sh
bash run.sh [path_to_test.jsonl] [path_to_prediction.jsonl]
```


## Usage
### Use the Script
```
usage: eval.py [-h] [-r REFERENCE] [-s SUBMISSION]

optional arguments:
  -h, --help            show this help message and exit
  -r REFERENCE, --reference REFERENCE
  -s SUBMISSION, --submission SUBMISSION
```

Example:
```
python eval.py -r public.jsonl -s submission.jsonl
{
  "rouge-1": {
    "f": 0.21999419163162043,
    "p": 0.2446195813913345,
    "r": 0.2137398792982201
  },
  "rouge-2": {
    "f": 0.0847583291303246,
    "p": 0.09419044877345074,
    "r": 0.08287844474014894
  },
  "rouge-l": {
    "f": 0.21017939117006337,
    "p": 0.25157090570020846,
    "r": 0.19404349000921203
  }
}
```


### Use Python Library
```
>>> from tw_rouge import get_rouge
>>> get_rouge('我是人', '我是一個人')
{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}
>>> get_rouge(['我是人'], [ '我是一個人'])
{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}
>>> get_rouge(['我是人'], ['我是一個人'], avg=False)
[{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}]
```


## Reference
[cccntu/tw_rouge](https://github.com/cccntu/tw_rouge)
