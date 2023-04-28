import ALS   
import argparse
import pandas as pd
from collections import defaultdict
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Recommendation task")
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--train_course_file", type=str, default=None, help="A csv or a json file containing the training course data."
    )
    parser.add_argument(
        "--course_file", type=str, default=None, help="A csv or a json file containing the course data."
    )
    parser.add_argument(
        "--subgroup_file", type=str, default=None, help="A csv or a json file containing the subgroup data."
    )
    parser.add_argument(
        "--user_file", type=str, default=None, help="A csv or a json file containing the user data."
    )
    args = parser.parse_args()
    return args

def construct_course_map(data_path, subgroup2id):
    df = pd.read_csv(data_path)
    course = {}
    for _, row in df.iterrows():
        course_id = row["course_id"]
        subgroups = row["sub_groups"].split(',') if row["sub_groups"] == row["sub_groups"] else []
        subgroups_id = []
        for subgroup in subgroups:
            subgroups_id.append(subgroup2id[subgroup])
        course[course_id] = subgroups_id
    return course

def construct_user_subgroup(data_path, course2subgroup):
    df = pd.read_csv(data_path)
    users = defaultdict(lambda: defaultdict(float))
    for _, row in df.iterrows():
        user_id = row["user_id"]
        courses_id = row["course_id"].split() if row["course_id"] == row["course_id"] else []
        for course_id in courses_id:
            subgroups = course2subgroup[course_id]
            for subgroup in subgroups:
                users[user_id][subgroup] += 2
    return users

def load_user_interest(data_path, users, subgroup2id):
    df = pd.read_csv(data_path)
    for _, row in df.iterrows():
        user_id = row["user_id"]
        if user_id in users.keys():
            interests = row["interests"].split(',') if row["interests"] == row["interests"] else []
            for interest in interests:
                interest = interest.split('_')[1]
                if interest in subgroup2id.keys():
                    users[user_id][subgroup2id[interest]] += 2

    return users

def main(args):
    label_df = pd.read_csv(args.subgroup_file)
    subgroup2id = {subgroup["subgroup_name"]: idx + 1 for idx, subgroup in label_df.iterrows()}
    course2subgroup = construct_course_map(args.course_file, subgroup2id)
    users = construct_user_subgroup(args.train_course_file, course2subgroup)
    users = load_user_interest(args.user_file, users, subgroup2id)

    def load_data(data_path, type="train"):
        df = pd.read_csv(data_path)
        data = []
        user_ids = []
        subgroups = []
        for _, row in df.iterrows():
            if type == "train" or type == "val":
                subgroup = [int(number) for number in row["subgroup"].split()] if row["subgroup"] == row["subgroup"] else []
                subgroups.append(subgroup)
            user_ids.append(row["user_id"])

        for user_id, user_subgroups in users.items():
            for user_subgroup in user_subgroups.items():
                data.append([user_subgroup[0], user_subgroup[1], user_id])

        return data, user_ids, subgroups

    raw_train, user_ids_train, subgroups_train = load_data(args.train_file)
    raw_val, user_ids_val, subgroups_val = load_data(args.validation_file, type="val")

    model = ALS.ALS()
    model.fit(data=raw_train, user_ids=user_ids_train, k=10, eval_data=(user_ids_val, subgroups_val), max_iter=10)
    save_file = "./model.pkl"
    with open(save_file, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)