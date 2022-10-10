from unittest.mock import inplace

import pandas as pd
import sklearn
import xgboost as xgb
from  xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_predict
from sklearn.metrics import accuracy_score
import numpy as np
import datetime as dt

train_df = pd.read_pickle('./data/sale_forcast/features_train.pkl')
test_df = pd.read_pickle('./data/sale_forcast/features_test.pkl')
train_df.reset_index(inplace=True)
category_df = pd.read_csv('./data/sale_forcast/category.csv')
for i_row in category_df.iterrows():
    cate_list = i_row[1].dropna().tolist()
    cate_name_str = cate_list[0]
    cate_list = cate_list[1:]
    if cate_name_str == 'weird_family_dat_miss':
        drop_idx = train_df.where((train_df['family'].isin(cate_list)) &\
                           (train_df['date']< dt.datetime.strptime('20160101', '%Y%m%d'))).dropna(axis=0).index
        train_df.drop(index=drop_idx,inplace=True)
    if cate_name_str == 'weird_miss':
        drop_idx = train_df.where((train_df['family'].isin(cate_list)) & \
                                  (train_df['date'] < dt.datetime.strptime('20161008', '%Y%m%d'))).dropna(axis=0).index
        train_df.drop(index=drop_idx, inplace=True)
def rmsle(y_true:np.array,y_pred:np.array)->list:
    y_pred = np.expm1(y_pred)
    y_pred[y_pred<-1] = -1 + 1e-6
    rmsle_float = np.sqrt(np.mean(np.power(np.log1p(y_pred)-np.log1p(y_true),2)))
    if np.isnan(rmsle_float):
        pass
    return ('rmsle',rmsle_float,False)

trainr_df = train_df.iloc[:,5:]
testr_df = test_df.iloc[:,2:]

param = {'max_depth':20, 'learning_rate':0.3,'n_estimators':500,'objective':'reg:squaredlogerror','eval_metric':'rmsle','alpha':0.3,'tree_method':'hist'}
model = xgb.XGBRegressor(**param)

trainr_array,val_array = train_test_split(trainr_df.values,test_size=int(0.2*len(trainr_df)),train_size=len(trainr_df)-int(0.2*len(trainr_df)))
bst = model.fit(trainr_array[:,1:],trainr_array[:,0])
#train dat
xgb.plot_importance(bst,height=0.9)
train_preds = np.expm1(bst.predict(trainr_df.iloc[:,1:].values))
train_preds[train_preds<-1]=-1+1e-6
train_accuracy = np.sqrt(np.mean((np.log((np.expm1(trainr_df.iloc[:,0].values)+1)/(train_preds+1))**2)))
print(train_accuracy)
#test dat
test_preds = np.expm1(bst.predict(testr_df.iloc[:,1:].values))
test_preds[test_preds<-1]=-1+1e-6
#test_accuracy = np.sqrt(np.mean((np.log((l_test+1)/(test_preds+1))**2)))
#print(test_accuracy)
submission_df = pd.read_csv('./data/sale_forcast/sample_submission.csv')
submission_df.iloc[:,1] = test_preds
'''zero_df = pd.read_csv('./data/sale_forcast/zero_forcast.csv')
zero_df = zero_df[zero_df['0']==0]
submission_df.iloc[zero_df.iloc[:,0].values,1] = 0'''
submission_df.set_index('id',inplace=True)
submission_df.to_csv('./data/sale_forcast/submission_est100_xgb.csv')
pass