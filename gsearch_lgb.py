import pandas as pd
import sklearn
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.model_selection import KFold,cross_val_predict
from sklearn.metrics import accuracy_score
import numpy as np
import lightgbm
from typing import Tuple
train_df = pd.read_pickle('./data/sale_forcast/features_train.pkl')
test_df = pd.read_pickle('./data/sale_forcast/features_test.pkl')


trainr_df = train_df.iloc[:,1:]
testr_df = test_df.iloc[:,1:]
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
l_train  = trainr_df.iloc[:,0].values

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
    predt[predt < -1] = -1 + 1e-6
    grad = gradient(predt, y_true)
    hess = hessian(predt, y_true)
    return grad, hess

def rmsle(y_true:np.array,y_pred:np.array)->list:
    y_pred[y_pred<-1] = -1 + 1e-6
    rmsle_float = np.sqrt(np.mean(np.power(np.log1p(y_pred)-np.log1p(y_true),2)))
    return ('rmsle',rmsle_float,False)

param_lgb = {'boosting_type':'goss','num_leaves':100,'max_depth':-1,'subsample':0.8,'learning_rate':0.05,\
             'n_estimators':500,'n_jobs':-1,'reg_alpha':0.2,'colsample_bytree':0.8,'objective':squared_log,'metric':'custom',\
             'min_child_samples':10}
params_lgb = {'num_leaves':np.linspace(50,2000,200,dtype=np.int32).tolist()}

modellgb = lightgbm.LGBMRegressor(**param_lgb)
trainr_array,val_array = train_test_split(trainr_df.values,test_size=int(0.2*len(trainr_df)),train_size=len(trainr_df)-int(0.2*len(trainr_df)))

rsearch = RandomizedSearchCV(modellgb,params_lgb, scoring='neg_mean_squared_error', cv=10,n_jobs=1,pre_dispatch=1)
bst = rsearch.fit(trainr_array[:,1:],trainr_array[:,0],eval_set = [(val_array[:,1:],val_array[:,0])],\
                  eval_metric=rmsle,callbacks=[lightgbm.early_stopping(stopping_rounds=5)])

print("Best score: %0.3f" % rsearch.best_score_)
print("Best parameters set:")
best_parameters = rsearch.best_estimator_.get_params()
for param_name in sorted(params_lgb.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))