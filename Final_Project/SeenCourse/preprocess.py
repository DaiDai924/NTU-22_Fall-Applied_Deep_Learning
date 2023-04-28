
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime

from dataset import HahowDataset


def load_dataset(args):
    train_dataset = pd.read_csv(args.train_file)
    eval_dataset = pd.read_csv(args.eval_file)
    test_dataset = pd.read_csv(args.test_file)
    courses = pd.read_csv(args.courses_file)

    lookup = build_id_lookup(train_dataset)
    lookup['course_id2attr'] = build_course_id2attr(courses)
    lookup['user_idx2done_courses'] = build_done_courses_dict(train_dataset, eval_dataset, lookup)

    train_dataset = HahowDataset(args, train_dataset, lookup)
    eval_dataset = HahowDataset(args, eval_dataset, lookup, True, train_dataset.data)
    test_dataset = HahowDataset(args, test_dataset, lookup, True, train_dataset.data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    return train_dataloader, eval_dataloader, test_dataloader, lookup


def build_id_lookup(train_dataset: pd.DataFrame) -> dict:
    lookup = {
        'user_id2idx': {},
        'user_idx2id': {},
        'course_id2idx': {},
        'course_idx2id': {},
    }

    user_ids = []
    course_ids = set()
    for i, row in train_dataset.iterrows():
        user_ids.append(row.user_id)
        course_ids.update(row.course_id.split())
    print(f'user #: {len(user_ids)}')
    print(f'course #: {len(course_ids)}')
    assert len(user_ids) == len(np.unique(user_ids))
    course_ids = sorted(course_ids)

    for i, user_id in enumerate(user_ids):
        lookup['user_id2idx'][user_id] = i
        lookup['user_idx2id'][i] = user_id
    for i, course_id in enumerate(course_ids):
        lookup['course_id2idx'][course_id] = i
        lookup['course_idx2id'][i] = course_id

    return lookup


def build_attr_lookup(courses: pd.DataFrame):
    course_lookup = {
        'teacher_id2idx': {},
        'group_id2idx': {},
        'subgroup_id2idx': {},
    }
    # ! group
    course_groups = set()
    for group_str in courses.groups:
        if isinstance(group_str, str):
            for group in group_str.split(','):
                course_groups.add(group)
    for i, group in enumerate(['nan'] + sorted(course_groups)):
        course_lookup['group_id2idx'][group] = i

    # ! subgroup
    course_subgroups = set()
    for subgroup_str in courses.sub_groups:
        if isinstance(subgroup_str, str):
            for subgroup in subgroup_str.split(','):
                course_subgroups.add(subgroup)
    for i, subgroup in enumerate(['nan'] + sorted(course_subgroups)):
        course_lookup['subgroup_id2idx'][subgroup] = i

    # ! teacher
    for i, teacher_id in enumerate(courses.teacher_id.unique()):
        course_lookup['teacher_id2idx'][teacher_id] = i
    return course_lookup


def course_process(courses: pd.DataFrame):
    attr_lookup = build_attr_lookup(courses)

    courses['time'] = 0
    courses['teacher_idx'] = 0
    for i, row in courses.iterrows():

        # ! time
        time_str = str(row.course_published_at_local)
        timestamp = int(datetime.strptime(time_str.split()[0], '%Y-%m-%d').timestamp()) if time_str != 'nan' else 0
        courses.iloc[i, courses.columns.get_loc('time')] = timestamp / 100

        courses.iloc[i, courses.columns.get_loc('teacher_idx')] = attr_lookup['teacher_id2idx'].get(row.teacher_id, -1)

        # ! n-hot group
        groups = [0] * len(attr_lookup['group_id2idx'])
        for group in str(row.groups).split(','):
            groups[attr_lookup['group_id2idx'][group]] = 1
        courses.iloc[i, courses.columns.get_loc('groups')] = ' '.join([str(x) for x in groups])

        # ! n-hot subgroup
        subgroups = [0] * len(attr_lookup['subgroup_id2idx'])
        for subgroup in str(row.sub_groups).split(','):
            subgroups[attr_lookup['subgroup_id2idx'][subgroup]] = 1
        courses.iloc[i, courses.columns.get_loc('sub_groups')] = ' '.join([str(x) for x in subgroups])

    time_min = courses.loc[courses.time != 0].time.min()
    time_max = courses.loc[courses.time != 0].time.max()
    courses.time = (courses.time-time_min) / (time_max-time_min)

    price_min = courses.course_price.min()
    price_max = courses.course_price.max()
    courses.course_price = (courses.course_price-price_min) / (price_max-price_min)

    courses.set_index('course_id', inplace=True)
    courses_filtered = courses[['course_price', 'teacher_idx', 'time', 'groups', 'sub_groups']]

    return courses_filtered


def build_course_id2attr(courses: pd.DataFrame):
    courses_filtered = course_process(courses)
    id2attr = {}
    for course_id in courses_filtered.index:
        attrs = []
        for i, item in courses_filtered.loc[course_id].items():
            attrs += [float(x) for x in str(item).split()]
        id2attr[course_id] = attrs
    return id2attr


def build_done_courses_dict(train_dataset: pd.DataFrame, eval_dataset: pd.DataFrame, lookup: dict):
    done_courses_dict = {}

    for i, row in train_dataset.iterrows():
        done_courses_dict[row.user_id] = row.course_id.split()
    for i, row in eval_dataset.iterrows():
        done_courses_dict[row.user_id] += row.course_id.split()

    done_courses_idx_dict = {}
    for user_id in done_courses_dict.keys():
        user_idx = lookup['user_id2idx'][user_id]
        done_courses_idx_dict[user_idx] = [lookup['course_id2idx'].get(course_id, -1) for course_id in done_courses_dict[user_id]]

    return done_courses_idx_dict
