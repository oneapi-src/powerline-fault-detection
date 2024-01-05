# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
This code has the functions needed for train-test-split, model training, and batch/streaming inference
'''
import time
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV  # pylint: disable=C0415
from sklearn.ensemble import RandomForestClassifier  # pylint: disable=C0415
from sklearn import metrics  # pylint: disable=C0415
from sklearnex import patch_sklearn  # pylint: disable=E0401,C0415
patch_sklearn()


def train_model(data, streaming_flag):
    X = data[[
            'Signal_to_Noise_Ratio', 'Num_of_Positive_True_Peaks', 'Num_of_Negative_True_Peaks', 'Std_Dev_of_True_Peak_Positions',
            'Mean_of_Positive_True_Peak_Heights', 'Min_of_Positive_True_Peak_Heights', 'Max_of_Positive_True_Peak_Heights',
            'Std_Dev_of_Positive_True_Peak_Heights', 'Mean_of_Negative_True_Peak_Heights', 'Min_of_Negative_True_Peak_Heights',
            'Max_of_Negative_True_Peak_Heights', 'Std_Dev_of_Negative_True_Peak_Heights', 'Mean_of_True_Peak_Widths',
            'Min_of_True_Peak_Widths', 'Max_of_True_Peak_Widths', 'Std_Dev_of_True_Peak_Widths']]

    y = data['Target']

    # Hyperparameters
    params = {
        'n_estimators': [100, 500, 1000],
        'max_leaf_nodes': [None, 10, 20, 30],
        'max_features': [None, 'sqrt']
        }

    bench_dict = {}

    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    bench_dict["Split"] = time.time()-start_time

    start_time = time.time()
    model_rf = GridSearchCV(RandomForestClassifier(
            criterion='gini', max_depth=None, n_jobs=-1, oob_score=True, random_state=42), param_grid=params, cv=5)

    model_rf.fit(X_train, y_train)
    bench_dict["Train"] = time.time()-start_time

    model_rf = model_rf.best_estimator_

    if streaming_flag:
        individual_stream_times = []
        preds = {}
        for i in range(1000):  # pylint: disable=C0415,W0612
            sample_df = X_test.sample(n=1)
            start_time = time.time()
            y_pred = model_rf.predict(sample_df)
            individual_stream_times.append(time.time()-start_time)
            preds[str(sample_df.index[0])] = y_pred[0]
        avg_stream_time = sum(individual_stream_times) / len(individual_stream_times)
        bench_dict["Inference"] = avg_stream_time

    else:
        start_time = time.time()
        y_pred = model_rf.predict(X_test)
        bench_dict["Inference"] = time.time()-start_time
        bench_dict["Accuracy"] = metrics.accuracy_score(y_test, y_pred)
        bench_dict["F1"] = metrics.f1_score(y_test, y_pred, average='macro')
        preds = dict(zip(X_test.index, y_pred))

    return model_rf, pd.Series(preds), bench_dict
