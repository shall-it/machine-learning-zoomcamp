#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import sklearn
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction import DictVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
import xgboost as xgb


def load_data():
    data_url = 'https://raw.githubusercontent.com/shall-it/machine-learning-zoomcamp/refs/heads/main/07-midterm-project/digital-lifestyle.csv'
    df = pd.read_csv(data_url)

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for cat_col in categorical_columns:
        df[cat_col] = df[cat_col].str.lower().str.replace(' ', '_')

    return df


# def train_model(df):

#     categorical_features = list(df.dtypes[df.dtypes == 'object'].index)

#     numerical = list(df.select_dtypes(include=['int64', 'float64']).columns)
#     numerical_features = [col for col in numerical if col != 'high_risk_flag']

    
#     df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
#     df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

#     df_train = df_train.reset_index(drop=True)
#     df_val = df_val.reset_index(drop=True)
#     df_test = df_test.reset_index(drop=True)

#     y_train = df_train.high_risk_flag.values
#     y_val = df_val.high_risk_flag.values
#     y_test = df_test.high_risk_flag.values

#     del df_train['high_risk_flag']
#     del df_val['high_risk_flag']
#     del df_test['high_risk_flag']

#     train_dict = df_train[categorical_features + numerical_features].to_dict(orient='records')
#     val_dict = df_val[categorical_features + numerical_features].to_dict(orient='records')

#     C_best = 1

#     pipeline = make_pipeline(
#         DictVectorizer(sparse=False),
#         LogisticRegression(solver='liblinear', C=C_best, max_iter=1000, random_state=42)
#     )

#     pipeline.fit(train_dict, y_train)

#     y_pred = pipeline.predict_proba(val_dict)[:, 1]
#     convert_decision = (y_pred >= 0.5)
#     accuracy = (y_val == convert_decision).mean().round(3)

#     print('Accuracy is', accuracy)

#     return pipeline


def train_model_xgb(df):

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

    dv = DictVectorizer(sparse=False)

    train_dict = df_train[categorical_features + numerical_features].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    val_dict = df_val[categorical_features + numerical_features].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    features = list(dv.get_feature_names_out())
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
    
    xgb_params = {
        'eta': 0.3, 
        'max_depth': 6,
        'min_child_weight': 30,

        'objective': 'binary:logistic',
        'eval_metric': 'auc',

        'nthread': 8,
        'seed': 42,
        'verbosity': 1,
    }

    model = xgb.train(xgb_params, dtrain, num_boost_round=35)

    y_pred = model.predict(dval)
    auc = roc_auc_score(y_val, y_pred).round(3)

    print('Validation AUC is', auc)

    df_full_train = df_full_train.reset_index(drop=True)

    y_full_train = df_full_train.high_risk_flag.values

    del df_full_train['high_risk_flag']

    dv = DictVectorizer(sparse=False)

    dicts_full_train = df_full_train.to_dict(orient='records')    
    X_full_train = dv.fit_transform(dicts_full_train)

    dicts_test = df_test.to_dict(orient='records')
    X_test = dv.transform(dicts_test)

    features = list(dv.get_feature_names_out())

    dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)

    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

    model = xgb.train(xgb_params, dfulltrain, num_boost_round=35)

    y_pred = model.predict(dtest)
    auc = roc_auc_score(y_test, y_pred).round(3)

    print('Full Test AUC is', auc)

    return model, dv


def save_model(filename, model, dv):
    with open (filename, 'wb') as f_out:
        pickle.dump((model, dv), f_out)
    print(f'Model was saved to {filename}')


df = load_data()
# pipeline = train_model(df)
# save_model('model.bin', pipeline)
model, dv = train_model_xgb(df)
save_model('model.bin', model, dv)
