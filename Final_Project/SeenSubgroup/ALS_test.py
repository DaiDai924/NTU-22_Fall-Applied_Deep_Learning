import ALS   
import argparse
import pandas as pd
import pickle
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Recommendation task")
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--pred_file", type=Path, default="output.csv", help="path to save prediction result"
    )
    args = parser.parse_args()
    return args

def main(args):
    def load_data(data_path, type="train"):
        df = pd.read_csv(data_path)
        user_ids = []
        for _, row in df.iterrows():
            user_ids.append(row["user_id"])
        return user_ids
    
    user_ids_test = load_data(args.test_file, type="test")

    load_file = "./best_model.pkl"
    with open(load_file, 'rb') as f:
        model = pickle.load(f)
    
    prediction = model.predict(user_ids=user_ids_test, n_subgroups=50)
    pred = []
    for i, user_id in enumerate(user_ids_test):
        ans = ""
        for (subgroup, _) in prediction[i]:
            ans += str(subgroup) + " "
        pred.append([user_id, ans.strip()])

    df = pd.DataFrame(pred)
    df.to_csv(args.pred_file, header=["user_id","subgroup"], index=False)

if __name__ == "__main__":
    args = parse_args()
    args.pred_file.parent.mkdir(parents=True, exist_ok=True)
    main(args)