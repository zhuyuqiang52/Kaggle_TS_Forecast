from unittest.mock import inplace

import pandas as pd
import sklearn
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import accuracy_score
import numpy as np
import statsmodels.api as sm
import datetime as dt

train_df = pd.read_pickle('./data/sale_forcast/features_train.pkl')
test_df = pd.read_pickle('./data/sale_forcast/features_test.pkl')

def gradient(y_true: np.ndarray, predt: np.ndarray) -> np.ndarray:
    '''Compute the gradient squared log error.'''
    predt[predt < -1] = -1 + 1e-6
    return (np.log1p(predt) - np.log1p(y_true)) / (predt + 1)

def hessian(y_true: np.ndarray, predt: np.ndarray) -> np.ndarray:
    '''Compute the hessian for squared log error.'''
    predt[predt < -1] = -1 + 1e-6
    hes = ((-np.log1p(predt) + np.log1p(y_true) + 1) /
            np.power(predt + 1, 2))
    return hes

def squared_log(predt: np.ndarray,y_true: np.ndarray):
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    predt[predt < -1] = -1 + 1e-6
    grad = gradient(predt, y_true)
    hess = hessian(predt, y_true)
    return grad, hess

def rmsle(y_true: np.array, y_pred: np.array) -> list:
    y_pred[y_pred < -1] = -1 + 1e-6
    rmsle_float = np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))
    return rmsle_float


trainr_df = train_df.iloc[:, 1:].reset_index()
testr_df = test_df.iloc[:, 1:].reset_index()
trainr_df.drop(['Local','Regional','rank*Local','rank*Regional'],axis=1,inplace=True)
testr_df.drop(['Local','Regional','rank*Local','rank*Regional'],axis=1,inplace=True)
# category
category_df = pd.read_csv('./data/sale_forcast/category.csv')
submission_df = pd.read_csv('./data/sale_forcast/sample_submission.csv')
param = {'max_depth': 20, 'learning_rate': 0.2, 'n_estimators': 1000, 'objective': squared_log,
         'eval_metric': rmsle, 'alpha': 0.2}
for i_row in category_df.iterrows():
    cate_list = i_row[1].dropna().tolist()
    cate_name_str = cate_list[0]
    cate_list = cate_list[1:]
    train_tmp_df = trainr_df.loc[trainr_df['family'].isin(cate_list), :]
    if cate_name_str == 'weird_family_dat_miss':
        train_tmp_df = train_tmp_df[train_tmp_df['date'] > dt.datetime.strptime('20160101', '%Y%m%d')]
    if cate_name_str == 'weird_miss':
        train_tmp_df = train_tmp_df[train_tmp_df['date'] >= dt.datetime.strptime('20161008', '%Y%m%d')]
    train_tmp_df = train_tmp_df.iloc[:, 3:]
    test_tmp_df = testr_df.loc[testr_df['family'].isin(cate_list), :].iloc[:, 3:]
    model = xgb.XGBRegressor(**param)
    trainr_array, val_array = train_test_split(train_tmp_df.values, test_size=int(0.2 * len(train_tmp_df)), \
                                               train_size=len(train_tmp_df) - int(0.2 * len(train_tmp_df)))
    bst = model.fit(trainr_array[:, 1:], trainr_array[:, 0], eval_set=[(val_array[:, 1:], val_array[:, 0])])

    # train dat
    train_preds = bst.predict(train_tmp_df.iloc[:, 1:].values)
    train_preds[train_preds < -1] = -1 + 1e-6
    train_accuracy = np.sqrt(np.mean((np.log((train_tmp_df.iloc[:, 0].values + 1) / (train_preds + 1)) ** 2)))
    print(f'category{cate_name_str}:{train_accuracy}')

    importance_list = model.feature_importances_
    importance_sorted_list = (np.argsort(importance_list)[-20:]+1).tolist()
    model = xgb.XGBRegressor(**param)
    trainr_array, val_array = train_test_split(train_tmp_df.values, test_size=int(0.2 * len(train_tmp_df)), \
                                               train_size=len(train_tmp_df) - int(0.2 * len(train_tmp_df)))
    bst = model.fit(trainr_array[:, importance_sorted_list], trainr_array[:, 0], eval_set=[(val_array[:, importance_sorted_list], val_array[:, 0])])

    # train dat
    train_preds = bst.predict(train_tmp_df.iloc[:, importance_sorted_list].values)
    train_preds[train_preds < -1] = -1 + 1e-6
    train_accuracy = np.sqrt(np.mean((np.log((train_tmp_df.iloc[:, 0].values + 1) / (train_preds + 1)) ** 2)))
    print(f'category(feature selected){cate_name_str}:{train_accuracy}')
    # test dat
    test_preds = bst.predict(test_tmp_df.iloc[:, importance_sorted_list].values)
    test_preds[test_preds < -1] = -1 + 1e-6
    submission_df.loc[test_tmp_df.index, 'sales'] = test_preds

# zero_df = pd.read_csv('./data/sale_forcast/zero_forcast.csv')
# zero_df = zero_df[zero_df['0']==0]
# submission_df.iloc[zero_df.iloc[:,0].values,1] = 0
submission_df.set_index('id', inplace=True)
submission_df.to_csv('./data/sale_forcast/submission_est100.csv')
pass
