# unzip bestmodel zip file
# run test_summarization for the final predicted csv

python ./test_summarization.py \
    --model_name_or_path ./bestmodel/ \
    --test_file "${1}" \
    --source_prefix "summarize: " \
    --text_column maintext \
    --pred_file "${2}" \
    --num_beams 5 \
    --early_stopping True