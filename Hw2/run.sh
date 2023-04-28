# unzip bestmodel zip file
# run test_mc and test_qa for the final predicted csv

unzip bestmodel.zip

python ./test_mc.py \
        --model_name_or_path ./bestmodel/mc/ \
        --context_file "${1}" \
        --test_file "${2}" \
        --pred_file ./pred_relevant.npy

python ./test_qa.py \
        --model_name_or_path ./bestmodel/qa/ \
        --context_file "${1}" \
        --test_file "${2}" \
        --relevant_file ./pred_relevant.npy \
        --pred_file "${3}"