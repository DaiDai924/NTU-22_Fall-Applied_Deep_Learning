from argparse import ArgumentParser, Namespace
from pathlib import Path
import logging
import pickle
from tqdm import tqdm
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
import pandas as pd
import string
import zhon.hanzi
import re

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main(args):

    ######################### Reading data #########################

    user_file = args.data_dir / "users.csv"
    logging.info(f'Reading {user_file}')
    raw_users = pd.read_csv(user_file)
    raw_users.drop(['gender'], axis=1, inplace=True)

    course_file = args.data_dir / "courses.csv"
    logging.info(f'Reading {course_file}')
    raw_courses = pd.read_csv(course_file)
    raw_courses.drop(['course_price', 'teacher_id', 'teacher_intro', 'course_published_at_local', 'will_learn', 'required_tools', 'recommended_background', 'target_group'], axis=1, inplace=True)

    train_file = args.data_dir / "train.csv"
    logging.info(f'Reading {train_file}')
    raw_train = pd.read_csv(train_file)

    val_file = args.data_dir / "val_unseen.csv"
    logging.info(f'Reading {val_file}')
    raw_val_unseen = pd.read_csv(val_file)

    test_unseen_file = args.data_dir / "test_unseen.csv"
    logging.info(f'Reading {test_unseen_file}')
    raw_test_unseen = pd.read_csv(test_unseen_file)

    raw_data = {'course':raw_courses, 'train':raw_train, 'val':raw_val_unseen, 'test':raw_test_unseen}

    ######################### Data cleansing #########################

    def doc_clean(document):
        new_document = []
        all_punc = string.punctuation + zhon.hanzi.punctuation
        for doc in tqdm(document):
            doc = re.sub('[\d]', '', doc)
            doc = re.sub('[\s]', '', doc)
            doc = re.sub('[{}]'.format(all_punc), '', doc)
            doc = re.sub('[a-zA-Z]', '', doc)
            new_document.append(doc.strip())
        return new_document

    def query_clean(query):
        new_query = []
        all_punc = string.punctuation + zhon.hanzi.punctuation
        for q in tqdm(query):
            q = re.sub('[{}]'.format(all_punc), ' ', q)
            q = q.strip().split()
            q = set(q)
            q = ' '.join(q)
            new_query.append(q)
        return new_query

    def ws_clean(sentence_ws):
        short_sentence = []
        for word_ws in sentence_ws:
            per_sentence = []
            for w_ws in word_ws:
                # Get rid of one-character word
                is_not_one_character = not (len(w_ws) == 1)

                if is_not_one_character:
                    per_sentence.append(w_ws)
            short_sentence.append(per_sentence)

        return short_sentence
    
    def pos_clean(sentence_ws, sentence_pos):
        short_sentence = []
        for word_ws, word_pos in zip(sentence_ws, sentence_pos):
            per_sentence = []
            for w_ws, w_pos in zip(word_ws, word_pos):
                # Only keep nouns and verbs
                is_N_or_V = w_pos.startswith("V") or w_pos.startswith("N")

                # Get rid of one-character word
                is_not_one_character = not (len(w_ws) == 1)

                if is_N_or_V and is_not_one_character:
                    per_sentence.append(w_ws)
            short_sentence.append(per_sentence)

        return short_sentence

    ######################### Making query and document #########################

    data = {}
    for split, split_data in raw_data.items():
        logging.info(f'Making query and/or document of {split} dataset')
        if split == 'course':
            split_data = split_data.astype(str).replace('nan', '')
            split_data['document'] = split_data['course_name'].astype(str) + ',' + split_data['groups'].astype(str) + ',' + \
                split_data['sub_groups'].astype(str) + ',' + split_data['topics'].astype(str) + ',' + split_data['description'].astype(str)
            split_data.drop(['course_name', 'groups', 'sub_groups', 'topics', 'description'], axis=1, inplace=True)
            
            split_data['document'] = doc_clean(split_data['document'].to_list())
            split_data.to_csv('./hahow/data/clean_course.csv')
            
            if split_data['document'][split_data['document']==',,,'].empty:
                logging.info(f'No empty document in {split} dataset')

        else:
            split_data = pd.merge(split_data, raw_users, on='user_id')

            split_data = split_data.astype(str).replace('nan', '')
            split_data['query'] = split_data['interests'].astype(str) + ',' + split_data['recreation_names'].astype(str)
            split_data['query'] = query_clean(split_data['query'].to_list())
            split_data.drop(['occupation_titles', 'interests', 'recreation_names'], axis=1, inplace=True)
            
            if split_data['query'][split_data['query']==','].empty:
                logging.info(f'No empty query in {split} dataset')

        data[split] = split_data

    ######################### Tokenization #########################
    
    ws = CkipWordSegmenter(model="albert-base", device=0)
    pos = CkipPosTagger(model="albert-base", device=0)

    corpus = data['course']['document'].tolist()
    val_query = data['val']['query'].tolist()
    test_query = data['test']['query'].tolist()

    logging.info(f'Tokenizing corpus')
    tokenized_corpus = ws(corpus)
    pos_corpus = pos(corpus)
    corpus_clean = pos_clean(tokenized_corpus, pos_corpus)
    path = args.pkl_dir / "tokenized_corpus_posclean.pkl"
    with open(path, "wb") as fp:
        pickle.dump(corpus_clean, fp)
    logging.info(f"Tokenized corpus saved at {str(path.resolve())}")

    logging.info(f'Tokenizing validation query')
    tokenized_val_query = ws(val_query)
    tokenized_val_query_clean = ws_clean(tokenized_val_query)
    path = args.pkl_dir / "tokenized_val_query_clean.pkl"
    with open(path, "wb") as fp:
        pickle.dump(tokenized_val_query_clean, fp)
    logging.info(f"Tokenized validation query saved at {str(path.resolve())}")

    logging.info(f'Tokenizing test query')
    tokenized_test_query = ws(test_query)
    tokenized_test_query_clean = ws_clean(tokenized_test_query)
    path = args.pkl_dir / "tokenized_test_query_clean.pkl"
    with open(path, "wb") as fp:
        pickle.dump(tokenized_test_query_clean, fp)
    logging.info(f"Tokenized test query saved at {str(path.resolve())}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--pkl_dir", type=Path, help="Directory to save the pkl file.", default="./pkl/")
    parser.add_argument("--data_dir", type=Path, default="./hahow/data/")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.pkl_dir.mkdir(parents=True, exist_ok=True)

    main(args)
