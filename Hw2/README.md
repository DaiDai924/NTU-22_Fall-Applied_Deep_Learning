# NTU ADL 2022 Fall Homework 2

## Environment
```shell
# use conda
make
pip install -r requirements.txt
```

## Context Selection (Multiple Choice)
### Train
```shell
python train_mc_no_trainer.py --context_file [context_file] --train_file [train_file] --validation_file [validation_file] --model_name_or_path [model_name_or_path] --max_length [max_length] --per_device_train_batch_size [per_device_train_batch_size] --per_device_eval_batch_size [per_device_eval_batch_size] --learning_rate [learning_rate] --num_train_epochs [num_train_epochs] --gradient_accumulation_steps [gradient_accumulation_steps] --output_dir [output_dir]
```

### Test
```shell
python test_mc.py --context_file [context_file] --test_file [test_file] --pred_file [pred_file] --model_name_or_path [model_name_or_path] --max_length [max_length] --per_device_eval_batch_size [per_device_eval_batch_size] --gradient_accumulation_steps [gradient_accumulation_steps]
```

## Question Answering
### Train
```shell
python train_qa.py --context_file [context_file] --train_file [train_file] --validation_file [validation_file] --model_name_or_path [model_name_or_path] --max_length [max_length] --per_device_train_batch_size [per_device_train_batch_size] --per_device_eval_batch_size [per_device_eval_batch_size] --learning_rate [learning_rate] --num_train_epochs [num_train_epochs] --gradient_accumulation_steps [gradient_accumulation_steps] --output_dir [output_dir] --do_train --do_eval --evaluation_strategy [steps, (no, epoch)] --eval_steps [eval_steps] --save_steps [save_steps] --save_total_limit [save_total_limit] --logging_steps [logging_steps] --warmup_ratio [warmup_ratio] --disable_tqdm [True/False] 
```

### Test
```shell
python test_qa.py --context_file [context_file] --test_file [test_file] --relevant_file [relevant_file] --pred_file [pred_file] --model_name_or_path [model_name_or_path] --max_length [max_length] --per_device_eval_batch_size [per_device_eval_batch_size] --gradient_accumulation_steps [gradient_accumulation_steps]
```

### Draw learning curves of QA 
```shell
python learning_curve.py --trainer_state_file [trainer_state_file] --figure_title [figure_title]
```

## Reproduce my result (Public: 0.81193, Private: 0.80578)
```shell
bash download.sh
bash run.sh [path_to_context.json] [path_to_test.json] [path_to_prediction.csv]
```
