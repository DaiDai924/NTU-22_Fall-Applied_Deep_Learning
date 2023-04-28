import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
import logging
import pickle
from tqdm import tqdm

import pandas as pd
pd.set_option('display.max_column', None)
pd.set_option('max_colwidth', 100)

from collections import Counter 

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main(args):

    ######################### Reading data #########################

    logging.info(f'Reading {args.users_file}')
    raw_users = pd.read_csv(args.users_file)
    raw_users.drop(['gender'], axis=1, inplace=True)

    raw_data = {}

    if args.do_train:
        logging.info(f'Reading {args.train_file}')
        raw_train = pd.read_csv(args.train_file)
        raw_data['train'] = raw_train

    if args.do_val:
        logging.info(f'Reading {args.val_file}')
        raw_val_unseen = pd.read_csv(args.val_file)
        raw_data['val'] = raw_val_unseen

    if args.do_predict:
        logging.info(f'Reading {args.test_file}')
        raw_test_unseen = pd.read_csv(args.test_file)
        raw_data['test'] = raw_test_unseen

    ######################### processing data #########################

    def query_clean(query):
        new_query = []
        for q in tqdm(query):
            q = q.strip(',').split(',')
            new_query.append(q)
        return new_query

    data = {}
    for split, split_data in raw_data.items():
        logging.info(f"Processing the {split} dataset.")

        if split == 'train':
            split_data = pd.merge(split_data, raw_users, on='user_id')
            split_data = split_data.astype(str).replace('nan', '')
            
            split_data['query'] = split_data['interests'].astype(str)
            split_data['query'] = query_clean(split_data['query'].to_list())
            split_data.drop(['user_id', 'occupation_titles', 'interests', 'recreation_names'], axis=1, inplace=True)

            split_data['course_id'] = split_data.course_id.str.split(' ')
        else:
            split_data = pd.merge(split_data, raw_users, on='user_id')
            split_data = split_data.astype(str).replace('nan', '')

            split_data['query'] = split_data['interests'].astype(str)
            split_data['query'] = query_clean(split_data['query'].to_list())
            split_data.drop(['occupation_titles', 'interests', 'recreation_names'], axis=1, inplace=True)
            
            split_data['course_id'] = split_data.course_id.str.split(' ')

        data[split] = split_data

    ######################### Training #########################

    if args.do_train:
        logging.info(f"Training")

        query2course = {}

        for query, course in zip(data['train']['query'], data['train']['course_id']):
            counter = Counter(course)
            for q in query:
                if q in query2course.keys():
                    query2course[q] = query2course[q] + counter
                else:
                    query2course[q] = counter

        with open("query2course.pkl", "wb") as fp:
            pickle.dump(query2course, fp)
        logging.info(f"query2course saved at ./query2course.pkl")

    ######################### Evaluation #########################

    if(args.do_val):
        logging.info(f"Validation")

        with open("query2course.pkl", "rb") as fp:
            query2course = pickle.load(fp)
        logging.info(f"query2course query loaded at ./query2course.pkl")

        all_predictions = []
        for query in tqdm(data['val']['query']):
            candidates = Counter()
            for q in query:
                if q in query2course.keys():
                    candidates = candidates + query2course[q]

            pred = []
            for k, v in candidates.most_common(50):
                pred.append(k)

            pred = ' '.join(pred)
            all_predictions.append(pred)

        data['val']['course_id'] = all_predictions

        pred_dir = os.path.dirname(args.val_pred_file)
        if pred_dir != '' and not os.path.isdir(pred_dir):
            os.makedirs(pred_dir)
        data['val'][['user_id', 'course_id']].to_csv(args.val_pred_file, encoding='utf-8', index=False)
        logging.info(f"Evaluation results saved at {args.val_pred_file}")

    ######################### Predicting #########################

    if(args.do_predict):
        logging.info(f"Predicting")

        with open("query2course.pkl", "rb") as fp:
            query2course = pickle.load(fp)
        logging.info(f"query2course query loaded at ./query2course.pkl")

        all_predictions = []
        for query in tqdm(data['test']['query']):
            candidates = Counter()
            for q in query:
                if q in query2course.keys():
                    candidates = candidates + query2course[q]

            pred_course_id = []
            for k, v in candidates.most_common(50):
                pred_course_id.append(k)

            pred_course_id = ' '.join(pred_course_id)
            all_predictions.append(pred_course_id)

        data['test']['course_id'] = all_predictions

        pred_dir = os.path.dirname(args.test_pred_file)
        if pred_dir != '' and not os.path.isdir(pred_dir):
            os.makedirs(pred_dir)
        data['test'][['user_id', 'course_id']].to_csv(args.test_pred_file, encoding='utf-8', index=False)
        logging.info(f"Prediction results saved at {args.test_pred_file}")       



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--users_file", type=str, default="./data/users.csv")
    parser.add_argument("--train_file", type=str, default="./data/train.csv")
    parser.add_argument("--val_file", type=str, default="./data/val_unseen.csv")
    parser.add_argument("--test_file", type=str, default="./data/test_unseen.csv")
    parser.add_argument("--val_pred_file", type=str, default="./pred_val_unseen.csv")
    parser.add_argument("--test_pred_file", type=str, default="./pred_test_unseen.csv")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_val", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)