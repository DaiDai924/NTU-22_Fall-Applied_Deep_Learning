from argparse import ArgumentParser, Namespace
from pathlib import Path
import logging
import pickle
from tqdm import tqdm
from average_precision import mapk
import pandas as pd
import numpy as np

from rank_bm25 import BM25Okapi, BM25L, BM25Plus

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def load_subgroup_or_course(data_path, type="subgroup"):
    df = pd.read_csv(data_path)
    datas = []
    for _, row in df.iterrows():
        if type == "subgroup":
            data = [int(number) for number in row["subgroup"].split()] if row["subgroup"] == row["subgroup"] else []
        else:
            data = [number for number in row["course_id"].split()] if row["course_id"] == row["course_id"] else []
        datas.append(data)
    return datas

def construct_course_map(data_path, subgroup2id):
    df = pd.read_csv(data_path)
    course2subgroup = {}
    for _, row in df.iterrows():
        course_id = row["course_id"]
        subgroups = row["sub_groups"].split(',') if row["sub_groups"] == row["sub_groups"] else []
        subgroups_id = []
        for subgroup in subgroups:
            subgroups_id.append(subgroup2id[subgroup])
        course2subgroup[course_id] = subgroups_id
    return course2subgroup

def main(args):

    ######################### Reading data #########################

    course_file = args.data_dir / "courses.csv"
    logging.info(f'Reading {course_file}')
    raw_courses = pd.read_csv(course_file)
    raw_courses.drop(['course_price', 'teacher_id', 'teacher_intro', 'course_published_at_local', 'will_learn', 'required_tools', 'recommended_background', 'target_group'], axis=1, inplace=True)

    val_file = args.data_dir / "val_unseen.csv"
    logging.info(f'Reading {val_file}')
    raw_val_unseen = pd.read_csv(val_file)
    
    test_unseen_file = args.data_dir / "test_unseen.csv"
    logging.info(f'Reading {test_unseen_file}')
    raw_test_unseen = pd.read_csv(test_unseen_file)

    data = {'val': raw_val_unseen, 'test': raw_test_unseen, 'course': raw_courses}

    ######################### Construct course and subgroup mapping #########################

    subgroup_file = args.data_dir / "subgroups.csv"
    raw_subgroup = pd.read_csv(subgroup_file)
    subgroup2id = {subgroup["subgroup_name"]: idx + 1 for idx, subgroup in raw_subgroup.iterrows()}
    course2subgroup = construct_course_map(course_file, subgroup2id)

    ######################### Making query and document #########################

    ## Done by preprocess.py

    ######################### Tokenization #########################
    
    ## Done by preprocess.py

    ######################### Loading data and setting the algorithm #########################

    path = args.pkl_dir / "tokenized_corpus_posclean.pkl"
    with open(path, "rb") as fp:
        tokenized_corpus = pickle.load(fp)
    logging.info(f"Tokenized corpus loaded at {str(path.resolve())}")

    path = args.pkl_dir / "tokenized_val_query_clean.pkl"
    with open(path, "rb") as fp:
        tokenized_val_query = pickle.load(fp)
    logging.info(f"Tokenized validation query loaded at {str(path.resolve())}")

    path = args.pkl_dir / "tokenized_test_query_clean.pkl"
    with open(path, "rb") as fp:
        tokenized_test_query = pickle.load(fp)
    logging.info(f"Tokenized test query loaded at {str(path.resolve())}")

    if args.BM25Okapi:
        bm25 = BM25Okapi(tokenized_corpus)
    elif args.BM25L:
        bm25 = BM25L(tokenized_corpus)
    elif args.BM25Plus:
        bm25 = BM25Plus(tokenized_corpus)
    else:
        raise ValueError('You must specify the BM25 algorithm.')

    ######################### Evaluation #########################
    
    n_course = 150
    decay_weight = 0.95
    decay_step = 2

    if(args.eval):
        logging.info(f"Evaluation")

        pred = []
        for q in tqdm(tokenized_val_query):
            scores = bm25.get_scores(q)
            top_n = np.argsort(scores)[::-1][:n_course]
            pred_course_id = [data['course']['course_id'][i] for i in top_n]
            pred_course_id = ' '.join(pred_course_id)
            pred.append(pred_course_id)
        data['val']['course_id'] = pred

        data['val'][['user_id', 'course_id']].to_csv(args.val_pred_course_file, encoding='utf-8', index=False)
        logging.info(f"Evaluation results saved at {args.val_pred_course_file}")

        gt_path = args.data_dir / "val_unseen.csv"
        gt = load_subgroup_or_course(str(gt_path.resolve()), "course")
        p = load_subgroup_or_course(args.val_pred_course_file, "course")            
        print("MAPK: %.6f" % (mapk(gt, p, 50)))

    ######################### Evaluation subgroup #########################

        gt_path = args.data_dir / "val_unseen_group.csv"
        gt = load_subgroup_or_course(str(gt_path.resolve()))
        p = load_subgroup_or_course(args.val_pred_course_file, "course")

        pred_list = []
        for idx in range(len(p)):
            weight = 1.0
            subgroup_vote = {(k + 1): 0 for k in range(len(subgroup2id))}
            for i, course_id in enumerate(p[idx]):
                subgroup_ids = course2subgroup[course_id]

                for subg_id in subgroup_ids:
                    subgroup_vote[subg_id] += weight
                
                if i != 0 and i % decay_step == 0:
                    weight *= decay_weight

            subgroup_vote = {k: v for k, v in sorted(subgroup_vote.items(), key=lambda item: item[1], reverse=True)}
            pred_list.append(list(subgroup_vote.keys())[:50])
                
        print("MAPK: %.6f" % (mapk(gt, pred_list, 50)))
        
    ######################### Predicting #########################

    if(args.predict):
        logging.info(f"Predicting")

        pred = []
        for q in tqdm(tokenized_test_query):
            scores = bm25.get_scores(q)
            top_n = np.argsort(scores)[::-1][:n_course]
            pred_course_id = [data['course']['course_id'][i] for i in top_n]
            pred_course_id = ' '.join(pred_course_id)
            pred.append(pred_course_id)
        data['test']['course_id'] = pred

        data['test'][['user_id', 'course_id']].to_csv(args.test_pred_course_file, encoding='utf-8', index=False)
        logging.info(f"Prediction course results saved at {args.test_pred_course_file}")

    ######################### Predicting subgroup #########################

        p = load_subgroup_or_course(args.test_pred_course_file, "course")

        pred_list = []
        for idx in range(len(p)):
            weight = 1.0
            subgroup_vote = {(k + 1): 0 for k in range(len(subgroup2id))}
            for i, course_id in enumerate(p[idx]):
                subgroup_ids = course2subgroup[course_id]

                for subg_id in subgroup_ids:
                    subgroup_vote[subg_id] += weight
                
                if i != 0 and i % decay_step == 0:
                    weight *= decay_weight

            subgroup_vote = {k: v for k, v in sorted(subgroup_vote.items(), key=lambda item: item[1], reverse=True)}
            pred_list.append(list(subgroup_vote.keys())[:50])

        prediction = []
        for i, row in raw_test_unseen.iterrows():
            ans = ""
            for subgroup in pred_list[i]:
                ans += str(subgroup) + " "
            prediction.append([row["user_id"], ans.strip()])

        pred_subgroup = pd.DataFrame(prediction)
        pred_subgroup.to_csv(args.test_pred_subgroup_file, header=["user_id","subgroup"], index=False)
        logging.info(f"Prediction group results saved at {args.test_pred_subgroup_file}")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--pkl_dir", type=Path, help="Directory to save the pkl file.", default="./pkl/")
    parser.add_argument("--data_dir", type=Path, help="Directory of data file.", default="./hahow/data/")
    parser.add_argument("--val_pred_course_file", type=str, default="./pred.val_unseen_course.csv")
    parser.add_argument("--test_pred_course_file", type=str, default="./pred.test_unseen_course.csv")
    parser.add_argument("--test_pred_subgroup_file", type=str, default="./pred.test_unseen_subgroup.csv")
    parser.add_argument("--BM25Okapi", action="store_true")
    parser.add_argument("--BM25L", action="store_true")
    parser.add_argument("--BM25Plus", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)