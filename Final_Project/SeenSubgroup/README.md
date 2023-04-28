# ADL22-project-SEEN SUBGROUP
# Step for training models and testing

## Environment
```shell
pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

### Training
```shell
python ALS_train.py [-h] [--train_file TRAIN_DATA_FILE] [--validation_file EVAL_DATA_FILE] [--train_course_file TRAIN_COURSE_DATA_FILE] [--course_file COURSE_DATA_FILE] [--subgroup_file SUBGROUP_DATA_FILE] [--user_file USER_DATA_FILE]
```
```shell
E.g. 
python ALS_train.py \
--train_file ./hahow/data/train_group.csv \
--validation_file ./hahow/data/val_seen_group.csv \
--train_course_file ./hahow/data/train.csv \
--course_file ./hahow/data/courses.csv \
--subgroup_file ./hahow/data/subgroups.csv \
--user_file ./hahow/data/users.csv
```

### Testing
```shell
python ALS_test.py [-h] [--test_file TEST_DATA_FILE] [--pred_file PRED_FILE]
```
```shell
E.g.
python ALS_test.py \
--test_file ./hahow/data/test_seen_group.csv \
--pred_file ./pred/pred.csv
```