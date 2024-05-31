import pandas as pd
import numpy as np

def create_date_columns(df, date_col='iso_date', season_col='Sea', dateFormat="%d/%m/%Y"):
    df['iso_date'] = pd.to_datetime(df[date_col], format=dateFormat)
    df['start_year'] = df[season_col].apply(lambda x: int(x[:2]) + 2000)

def return_train_test_indices(df, season_col='start_year', date_col = 'iso_date'):
    dict_train = {}
    dict_test  = {}
    for s in df[season_col].unique():
        df_temp = df[df[season_col] <= s].copy()
        test_mask = (df_temp[season_col] == s) & (df_temp[date_col].dt.month == 4) & (df_temp[date_col].dt.day > 13)
        train_index = df_temp.index[~test_mask]
        test_index  = df_temp.index[test_mask]
        dict_train[s] = train_index
        dict_test[s]  = test_index
    return dict_train, dict_test

def return_train_test_dfs(df, season_col='start_year', date_col = 'iso_date'):
    dict_train = {}
    dict_test  = {}
    for s in df[season_col].unique():
        df_temp = df[df[season_col] <= s].copy()
        test_mask = (df_temp[season_col] == s) & (df_temp[date_col].dt.month == 4) & (df_temp[date_col].dt.day > 13)
        train = df_temp[~test_mask].copy()
        test = df_temp[test_mask].copy()
        dict_train[s] = train
        dict_test[s]  = test
    return dict_train, dict_test

def return_train_test_arrays(df, feature_colnames=['Lge'], target_colname='HS',  season_col='start_year', date_col = 'iso_date'):
    dict_train = {}
    dict_test  = {}
    for s in df[season_col].unique():
        df_temp = df[df[season_col] <= s].copy()
        test_mask = (df_temp[season_col] == s) & (df_temp[date_col].dt.month == 4) & (df_temp[date_col].dt.day > 13)
        train = df_temp[~test_mask].copy()
        X_train = train[feature_colnames].values
        Y_train = train[target_colname].values
        test = df_temp[test_mask].copy()
        X_test = test[feature_colnames].values
        Y_test = test[target_colname].values
        dict_train[s]['X'] = X_train
        dict_train[s]['Y'] = Y_train
        dict_test[s]['X'] = X_test
        dict_test[s]['Y'] = Y_test
    return dict_train, dict_test