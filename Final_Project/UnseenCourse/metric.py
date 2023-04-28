from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def apk(actual, predicted, k=50):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=50):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-r", "--reference", type=str, default="./data/val_unseen.csv")
    parser.add_argument("-s", "--submission", type=str, default="./pred_val_unseen.csv")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    logging.info(f'Reading {args.reference}')
    refs = pd.read_csv(args.reference)
    logging.info(f'Reading {args.submission}')
    preds = pd.read_csv(args.submission)

    refs['course_id'] = refs.course_id.str.split(' ')
    refs = refs['course_id'].tolist()
    preds['course_id'] = preds.course_id.str.split(' ')
    preds = preds['course_id'].tolist()

    logging.info(f'Calculating')
    score = mapk(refs, preds)
    logging.info(f'Score: {score}')
    
