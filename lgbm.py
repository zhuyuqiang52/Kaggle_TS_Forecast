import pandas as pd
import sklearn
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_predict
from sklearn.metrics import accuracy_score
import numpy as np
import lightgbm
from typing import Tuple
train_df = pd.read_pickle('./data/sale_forcast/features_train.pkl')
test_df = pd.read_pickle('./data/sale_forcast/features_test.pkl')


trainr_df = train_df.iloc[:,2:]
testr_df = test_df.iloc[:,2:]
feature_list = ['onpromotion', 'National', 'dcoilwtico', 'chg_ma_rate', 'rank', 'weekday', 'payday',
               'year', 'sales_delay1',
               'sales_delay7',
               'sales_delay30',
               'delay*payday',
               'delay1*weekday',
               'rank*National',
                'std180'
               ]
#dat_train,dat_test = train_test_split(dat_df,test_size=int(0.3*len(dat_df)),train_size=len(dat_df)-int(0.3*len(dat_df)))
l_train  = np.expm1(trainr_df.iloc[:,0].values)

def gradient(predt: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    '''Compute the gradient squared log error.'''
    return (np.log1p(predt) - np.log1p(y_true)) / (predt + 1)

def hessian(predt: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    '''Compute the hessian for squared log error.'''
    return ((-np.log1p(predt) + np.log1p(y_true) + 1) /
            np.power(predt + 1, 2))

def squared_log(y_true: np.ndarray,predt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    y_true = np.expm1(y_true)
    predt = np.expm1(predt)
    predt[predt <= -1] = -1 + 1e-6
    grad = gradient(predt, y_true)
    hess = hessian(predt, y_true)
    return grad, hess

def rmsle(y_true:np.array,y_pred:np.array)->list:
    y_true = np.expm1(y_true)
    y_exp_pred = np.expm1(y_pred)
    y_exp_pred[y_exp_pred<=-1] = -1 + 1e-6
    rmsle_float = np.sqrt(np.mean(np.power(np.log1p(y_exp_pred)-np.log1p(y_true),2)))
    return ('rmsle',rmsle_float,False)

lgb_params = {'metric': {'mse'},
              'boosting_type' : 'goss',
              'num_leaves': 200,
              'learning_rate': 0.1,
              #'feature_fraction': 0.8,
              'max_depth': 7,
              'verbose': 0,
              'num_boost_round': 5000,
              'early_stopping_rounds': 200,
              'nthread': -1,
             'force_col_wise':True}

num_round = 100
#modelxgb = xgb.XGBRegressor(**param_xgb)
#modellgb = lightgbm.LGBMRegressor(**param_lgb)
#bst = modelxgb.fit(trainr_df.iloc[:,1:].values,trainr_df.iloc[:,0].values)
trainr_array,val_array = \
    train_test_split(trainr_df.values,test_size=int(0.2*len(trainr_df)),train_size=len(trainr_df)-int(0.2*len(trainr_df)))
'''bst = modellgb.fit(trainr_array[:,1:],trainr_array[:,0],\
                   eval_set = [(val_array[:,1:],val_array[:,0])],eval_metric=rmsle)'''
#train dat
lgbtrain = lightgbm.Dataset(data=trainr_array[:,1:],label=trainr_array[:,0],feature_name=trainr_df.columns[1:].tolist())
lgbval = lightgbm.Dataset(data=val_array[:,1:],label=val_array[:,0],feature_name=trainr_df.columns[1:].tolist())
bst = lightgbm.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  #feval=mean_absolute_error,
                  verbose_eval=100,
                  )
train_preds = np.expm1(bst.predict(trainr_df.iloc[:,1:].values))
train_preds[train_preds<-1]=-1+1e-6
train_accuracy = np.sqrt(np.mean((np.log((np.expm1(trainr_df.iloc[:,0].values)+1)/(train_preds+1))**2)))
print(train_accuracy)
#test dat
test_preds = np.expm1(bst.predict(testr_df.iloc[:,1:].values))
test_preds[test_preds<0]=0.0
#test_accuracy = np.sqrt(np.mean((np.log((l_test+1)/(test_preds+1))**2)))
#print(test_accuracy)
submission_df = pd.read_csv('./data/sale_forcast/sample_submission.csv')
submission_df.iloc[:,1] = test_preds
'''zero_df = pd.read_csv('./data/sale_forcast/zero_forcast.csv')
zero_df = zero_df[zero_df['0']==0]'''
'''submission_df.iloc[zero_df.iloc[:,0].values,1] = 0'''
submission_df.set_index('id',inplace=True)
submission_df.to_csv('./data/sale_forcast/submission_lgb.csv')
pass