from unittest.mock import inplace

import pandas as pd
import sklearn
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
train_df = pd.read_pickle('./data/sale_forcast/festures_train.pkl')
test_df = pd.read_pickle('./data/sale_forcast/festures_test.pkl')

trainr_df = train_df.iloc[:, 1:]
testr_df = test_df.iloc[:, 1:]
# dat_train,dat_test = train_test_split(dat_df,test_size=int(0.3*len(dat_df)),train_size=len(dat_df)-int(0.3*len(dat_df)))
# train features
feature_list = ['onpromotion', 'Local', 'Regional', 'National', 'dcoilwtico', 'chg_ma_rate', 'rank', 'weekday', 'payday',
               'year', 'sales_delay1',
               'sales_delay7',
               'sales_delay30',
               'delay*payday',
               'delay1*weekday',
               'rank*Local',
               'rank*Regional',
               'rank*National',
               'typerank*Local',
               'typerank*Regional',
               'typerank*National',
               'clusterrank*Local',
               'clusterrank*Regional',
               'clusterrank*National'
               ]
l_train = trainr_df.iloc[:, 0].values
X_train = trainr_df.loc[:,feature_list].values
dat_train_dmat = xgb.DMatrix(trainr_df.loc[:, ].values, label=l_train)
# l_test  = testr_df.iloc[:,0].values
dat_test_dmat = xgb.DMatrix(testr_df.loc[:, feature_list].values)

param = {'max_depth': 10, 'learning_rate': 0.2, 'silent': 0, 'objective': 'reg:squaredlogerror', 'eval_metric': 'rmsle',
         'alpha': 0.2, 'n_estimators': 500}
params = {'max_depth': list(np.linspace(5, 30, 6,dtype=np.int32)),
          'learning_rate': list(np.linspace(0.1, 0.5, 5)),
          'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
          'alpha': np.linspace(0.1, 0.5, 5).tolist(),
          'n_estimators': np.linspace(100, 2000, 20,dtype=np.int32).tolist()}

model = xgb.XGBRegressor(**param)
gsearch = GridSearchCV(model, param_grid=params, scoring='neg_mean_squared_error', cv=10)
gsearch.fit(X_train,l_train)

print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(params.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))