from typing import List

import tensorflow as tf

from datetime import datetime, date
from holidays import country_holidays

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

list_of_holidays_in_poland = country_holidays('PL')

CASES_COLUMN_IDX = 0


def check_if_holiday(day: int, month: int, year: int) -> int:
    if date(year, month, day) in list_of_holidays_in_poland:
        return int(True)
    return int(False)


def scale_data(df, columns: List[str]):
    # df[columns] = Normalizer().fit_transform(df[columns])  # does not improve at all
    # df[columns] = MinMaxScaler().fit_transform(df[columns]) # worse
    df[columns] = StandardScaler().fit_transform(df[columns])  # better


def check_season(date_: datetime) -> str:
    spring_start = datetime(date_.year, 3, 21)
    summer_start = datetime(date_.year, 6, 22)
    autumn_start = datetime(date_.year, 9, 23)
    winter_start = datetime(date_.year, 12, 22)

    if date_ >= winter_start or date_ < spring_start:
        return 0  # "Winter"
    elif spring_start <= date_ < summer_start:
        return 1  # "Spring"
    elif summer_start <= date_ < autumn_start:
        return 2  # "Summer"
    elif autumn_start <= date_ < winter_start:
        return 3  # "Autumn"


def split_data(df, targets_column: str = 'target'):
    x = df.drop([targets_column], axis=1)
    y = df[targets_column].values
    return x, y


def add_target(df, cases_column: str = "cases", target_column: str = 'target'):
    df[target_column] = df[cases_column].shift(-1)


def add_windows(df: pd.DataFrame, config):
    step_features = []
    for step in range(1, config.sequence_length):
        d = pd.DataFrame()
        for column in df:
            d[f'{column}-{step}'] = df[column].shift(step)
        step_features.append(d)

    all = [df, *step_features]
    all_values = [df.values for df in all]

    _2d = pd.concat(all, axis=1).values  # for dataframe
    _3d = np.stack(all_values)

    return _2d, _3d


def remove_rows_with_nan(matrix, tensor, targets, config):
    """ when shifting target, there are nan rows created, this step removes them"""
    assert matrix.shape[0] == tensor.shape[1] == len(targets)
    matrix = matrix[config.sequence_length - 1:-1]
    tensor = tensor[:, config.sequence_length - 1:-1, :]
    targets = targets[config.sequence_length - 1:-1]
    return matrix, tensor, targets


def append_vector_to_tensor(tensor, vector):
    vector = np.expand_dims(vector, axis=0)
    repeated_vector = np.repeat(vector[:, np.newaxis, :], tensor.shape[0], axis=0)
    transposed_tensor = np.transpose(repeated_vector, (0, 2, 1))
    new_tensor = np.concatenate((tensor, transposed_tensor), axis=2)
    return new_tensor


def add_aggregates_2d(matrix, columns_number, config):
    cases_columns = [CASES_COLUMN_IDX + (columns_number * step) for step in range(config.sequence_length)]

    mean_values = np.mean(matrix[:, cases_columns], axis=1)
    min_values = np.min(matrix[:, cases_columns], axis=1)
    max_values = np.max(matrix[:, cases_columns], axis=1)
    median_values = np.median(matrix[:, cases_columns], axis=1)
    delta = max_values - min_values

    return np.c_[matrix, min_values, max_values, mean_values, median_values, delta]


def add_aggregates_3d(tensor):
    mean_values = np.mean(tensor[:, :, CASES_COLUMN_IDX], axis=0)
    min_values = np.min(tensor[:, :, CASES_COLUMN_IDX], axis=0)
    max_values = np.max(tensor[:, :, CASES_COLUMN_IDX], axis=0)
    median_values = np.median(tensor[:, :, CASES_COLUMN_IDX], axis=0)
    delta = max_values - min_values

    tensor = append_vector_to_tensor(tensor, mean_values)
    tensor = append_vector_to_tensor(tensor, min_values)
    tensor = append_vector_to_tensor(tensor, max_values)
    tensor = append_vector_to_tensor(tensor, median_values)
    tensor = append_vector_to_tensor(tensor, delta)
    return tensor


def preprocess_data(df: pd.DataFrame, config):
    df = df.sort_values("dateRep")  # you may remove this

    if config.additional_features:
        pass  # your feature engineering methods goes here

    columns_number = len(df.columns)  # do not move, important

    add_target(df)

    x, y = split_data(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=config.train_test_split.test_size,
                                                        random_state=config.train_test_split.random_state,
                                                        shuffle=False)

    if config.use_scaler:
        scale_data(x_train, columns=['cases'])
        scale_data(x_test, columns=['cases'])

    x_train_2d, x_train_3d = add_windows(x_train, config)
    x_test_2d, x_test_3d = add_windows(x_test, config)

    x_train_2d, x_train_3d, y_train = remove_rows_with_nan(x_train_2d, x_train_3d, y_train, config)
    x_test_2d, x_test_3d, y_test = remove_rows_with_nan(x_test_2d, x_test_3d, y_test, config)

    x_train_2d = add_aggregates_2d(x_train_2d, columns_number, config)
    x_test_2d = add_aggregates_2d(x_test_2d, columns_number, config)

    x_train_3d = add_aggregates_3d(x_train_3d)
    x_test_3d = add_aggregates_3d(x_test_3d)

    xtensor_train = tf.convert_to_tensor(np.transpose(x_train_3d, (1, 0, 2)), dtype=tf.float16)
    xtensor_test = tf.convert_to_tensor(np.transpose(x_test_3d, (1, 0, 2)), dtype=tf.float16)

    return x_train_2d, x_test_2d, xtensor_train, xtensor_test, y_train, y_test
