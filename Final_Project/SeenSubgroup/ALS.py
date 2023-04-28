## Citation & Reference of this code: https://github.com/mattzheng/pyALS

from collections import defaultdict
from random import random
import numpy as np
from average_precision import mapk

class ALS(object):
    def __init__(self) -> None:
        self.user_ids = None
        self.user_ids_dict = None
        self.subgroup_ids = None
        self.subgroup_ids_dict = None
        self.user_mat = None
        self.subgroup_mat = None
        self.shape = None
        self.rmse = None

    def preprocessing(self, data, user_ids):
        self.user_ids = np.array(list(set(user_ids)))
        self.user_ids_dict = dict(map(lambda x: x[::-1], enumerate(self.user_ids)))
        self.subgroup_ids = np.array([subgroup for subgroup in range(1, 92)])
        self.subgroup_ids_dict = dict(map(lambda x: x[::-1], enumerate(self.subgroup_ids)))        
        self.shape = (len(self.user_ids), len(self.subgroup_ids))

        ratings = defaultdict(lambda: defaultdict(float))
        ratings_T = defaultdict(lambda: defaultdict(float))
        for row in data:
            subgroup_id, rating, user_id = row
            ratings[user_id][subgroup_id] = rating
            ratings_T[subgroup_id][user_id] = rating

        return ratings, ratings_T

    def users_mul_ratings(self, users, ratings_T):
        def f(users_row, subgroup_id):
            user_ids = iter(ratings_T[subgroup_id].keys())
            scores = iter(ratings_T[subgroup_id].values())
            col_nos = map(lambda x: self.user_ids_dict[x], user_ids)
            _users_row = map(lambda x: users_row[x], col_nos)
            return sum(a * b for a, b in zip(_users_row, scores))

        ret = [[f(users_row, subgroup_id) for subgroup_id in self.subgroup_ids] for users_row in users]
        return np.array(ret)

    def subgroups_mul_ratings(self, subgroups, ratings):
        def f(subgroups_row, user_id):
            subgroup_ids = iter(ratings[user_id].keys())
            scores = iter(ratings[user_id].values())
            col_nos = map(lambda x: self.subgroup_ids_dict[x], subgroup_ids)
            _subgroups_row = map(lambda x: subgroups_row[x], col_nos)
            return sum(a * b for a, b in zip(_subgroups_row, scores))

        ret = [[f(subgroups_row, user_id) for user_id in self.user_ids] for subgroups_row in subgroups]
        return np.array(ret)

    def gen_random_matrix(self, n_rows, n_colums):
        data = [[random() for _ in range(n_colums)] for _ in range(n_rows)]
        return np.array(data)

    def get_rmse(self, ratings):
        m, n = self.shape
        mse = 0.0
        n_elements = sum(map(len, ratings.values()))
        for i in range(m):
            for j in range(n):
                user_id = self.user_ids[i]
                subgroup_id = self.subgroup_ids[j]
                rating = ratings[user_id][subgroup_id]
                if rating > 0:
                    user_row = self.user_mat[:, i].T
                    subgroup_col = self.subgroup_mat[:, j]
                    rating_hat = user_row @ subgroup_col
                    square_error = (rating - rating_hat) ** 2
                    mse += square_error / n_elements
        return mse ** 0.5

    def fit(self, data, user_ids, k, eval_data, max_iter=10):
        user_ids_val, subgroups_val = eval_data
        ratings, ratings_T = self.preprocessing(data, user_ids)
        self.user_subgroups = {k: set(v.keys()) for k, v in ratings.items()}
        m, n = self.shape

        # error_msg = "Parameter k must be less than the rank of original matrix"
        # assert k < min(m, n), error_msg

        self.user_mat = self.gen_random_matrix(k, m)

        for i in range(max_iter):
            if i % 2:
                subgroups = self.subgroup_mat
                self.user_mat = self.subgroups_mul_ratings(
                    np.linalg.inv(subgroups @ subgroups.T) @ subgroups, 
                    ratings
                )
            else:
                users = self.user_mat
                self.subgroup_mat = self.users_mul_ratings(
                    np.linalg.inv(users @ users.T) @ users,
                    ratings_T
                )
            rmse = self.get_rmse(ratings)

            prediction = self.predict(user_ids=user_ids_val, n_subgroups=50)
            pred_list = []
            for idx, _ in enumerate(user_ids_val):
                ans_list = []
                for (subgroup, _) in prediction[idx]:
                    ans_list.append(subgroup)
                pred_list.append(ans_list)

            print("Iterations: %d, RMSE: %.6f, MAPK: %.6f" % (i + 1, rmse, mapk(subgroups_val, pred_list, 50)))

        self.rmse = rmse

    def predict_subgroups(self, user_id, n_subgroups):
        users_col = self.user_mat[:, self.user_ids_dict[user_id]].T
        subgroups_col = enumerate(users_col @ self.subgroup_mat)
        subgroup_scores = map(lambda x: (self.subgroup_ids[x[0]], x[1]), subgroups_col)
        return sorted(subgroup_scores, key=lambda x: x[1], reverse=True)[:n_subgroups]

    def predict(self, user_ids, n_subgroups=10):
        return [self.predict_subgroups(user_id, n_subgroups) for user_id in user_ids]
