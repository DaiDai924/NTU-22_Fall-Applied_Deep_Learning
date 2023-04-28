args=(
    --batch_size 1024
    --max_len 32
    # --do_train
    --do_test
    --pretrained_file "ckpt/model_records_basic_32.pt"
    --record_file "records_basic_32.json"
    --num_epoch 100
)

python main.py "${args[@]}"