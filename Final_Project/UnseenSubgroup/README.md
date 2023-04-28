# ADL22-Final_Hahow (Unseen Subgroup)

## Environment
```shell
# install pytorch
pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

## Preprocess
```shell
python ./preprocess.py \
    --pkl_dir [path_to_pkl_dir]  \
    --data_dir [path_to_data_dir]
```

e.g. 
python ./preprocess.py \
    --pkl_dir ./pkl/ \
    --data_dir ./hahow/data/


### Test
```shell
python ./BM25.py \
    --pkl_dir [path_to_pkl_dir]  \
    --data_dir [path_to_data_dir] \
    --val_pred_course_file [path_to_pred_val_course] \ 
    --test_pred_course_file [path_to_pred_test_course] \ 
    --test_pred_subgroup_file [path_to_pred_test_subgroup] \ 
    [--BM25Okapi / --BM25L / BM25Plus] \ 
    [--eval] \ 
    [--predict]
```

e.g.
python ./preprocess.py \
    --pkl_dir ./pkl/ \
    --data_dir ./hahow/data/
    --val_pred_course_file ./pred.val_unseen_course.csv \ 
    --test_pred_course_file ./pred.test_unseen_course.csv \ 
    --test_pred_subgroup_file ./pred.val_unseen_subgroup.csv \ 
    --BM25Okapi \ 
    --predict

### Reproduce the result (Public: 0.29697, Private: 0.30758)

```shell
python ./BM25.py --BM25Okapi --predict --data_dir ./hahow/data/
```
