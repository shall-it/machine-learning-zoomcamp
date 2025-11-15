#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import sklearn
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def load_data():
    data_url = 'https://raw.githubusercontent.com/shall-it/machine-learning-zoomcamp/refs/heads/main/07-midterm-project/digital-lifestyle.csv'
    df = pd.read_csv(data_url)

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for cat_col in categorical_columns:
        df[cat_col] = df[cat_col].str.lower().str.replace(' ', '_')

    return df


def train_model(df):

    categorical_features = list(df.dtypes[df.dtypes == 'object'].index)

    numerical = list(df.select_dtypes(include=['int64', 'float64']).columns)
    numerical_features = [col for col in numerical if col != 'high_risk_flag']

    
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.high_risk_flag.values
    y_val = df_val.high_risk_flag.values
    y_test = df_test.high_risk_flag.values

    del df_train['high_risk_flag']
    del df_val['high_risk_flag']
    del df_test['high_risk_flag']

    train_dict = df_train[categorical_features + numerical_features].to_dict(orient='records')
    val_dict = df_val[categorical_features + numerical_features].to_dict(orient='records')

    C_best = 1

    pipeline = make_pipeline(
        DictVectorizer(sparse=False),
        LogisticRegression(solver='liblinear', C=C_best, max_iter=1000, random_state=42)
    )

    pipeline.fit(train_dict, y_train)

    y_pred = pipeline.predict_proba(val_dict)[:, 1]
    convert_decision = (y_pred >= 0.5)
    accuracy = (y_val == convert_decision).mean().round(3)

    print('Accuracy is', accuracy)

    return pipeline


def save_model(filename, model):
    with open (filename, 'wb') as f_out:
        pickle.dump(model, f_out)
    print(f'Model was saved to {filename}')


df = load_data()
pipeline = train_model(df)
save_model('model.bin', pipeline)

with open ('model.bin', 'wb') as f_out:
    pickle.dump(pipeline, f_out)
