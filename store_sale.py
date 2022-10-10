import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime as dt
import pymysql
from sqlalchemy import create_engine

#sql
engine = create_engine('mysql+pymysql://root:932230@localhost/sales_forcast?charset=utf8')
# read data
dat_path_str = './data/sale_forcast/'
train_r_df = pd.read_csv(dat_path_str + 'train.csv')
train_r_df['date'] = pd.to_datetime(train_r_df['date'])
#Liquor sales alter
liquor_df = train_r_df[train_r_df['family']=='LIQUOR,WINE,BEER']
liquor_df.sort_values(['store_nbr','date'],inplace=True)
liquor_df['sales'].replace(0.0,np.nan,inplace=True)
liquor_df['sales'].bfill(inplace=True)
train_r_df.loc[liquor_df.index,'sales'] = liquor_df['sales']
holiday_df = pd.read_csv(dat_path_str + 'holidays_events.csv')
holiday_df['date'] = pd.to_datetime(holiday_df['date'])
oil_df = pd.read_csv(dat_path_str + 'oil.csv')
oil_df['date'] = pd.to_datetime(oil_df['date'])
store_df = pd.read_csv(dat_path_str + 'stores.csv')
test_df = pd.read_csv(dat_path_str + 'test.csv')
test_df['date'] = pd.to_datetime(test_df['date'])
train_df = pd.concat([train_r_df,test_df],axis=0)
del train_r_df
#   holiday
holiday_locale_df = holiday_df.copy()
# filter effective holiday
# only early days in a long holiday lead to increase in sales
# 1.1 , though national holiday,lead to decrease in sales to 0
#sales increase as time goes by on a average level

#total sales
total_sale_df = train_df.pivot_table(index = ['date'],columns='family',values='sales',aggfunc='sum')
weird_family_list = ['BABY CARE','BOOKS','CELEBRATION','FROZEN FOODS','HOME AND KITCHEN I','HOME CARE',  'HOME AND KITCHEN II','LADIESWEAR','PLAYERS AND ELECTRONICS', 'PET SUPPLIES','MAGAZINES','LIQUOR,WINE,BEER','GROCERY II']
weird_family_pre_end_list = ['FROZEN FOODS','GROCERY II','LIQUOR,WINE,BEER']
weird_family_dat_miss_list = ['BABY CARE','HOME AND KITCHEN I','CELEBRATION','HOME CARE','MAGAZINES','HOME AND KITCHEN II','LADIESWEAR',\
                              'PLAYERS AND ELECTRONICS','PET SUPPLIES']

weird_miss_list = ['BOOKS']
others_list = total_sale_df.columns.tolist()
others_list = np.setdiff1d(others_list,weird_family_list).tolist()

category_df = pd.DataFrame([weird_family_dat_miss_list,weird_family_pre_end_list,weird_miss_list,others_list])
category_df.index = ['weird_family_dat_miss','weird_family_pre_end','weird_miss','Normal']
category_df.to_csv('./data/sale_forcast/category.csv')
holiday_locale_df = holiday_locale_df.loc[((holiday_locale_df['date'].dt.month==1)&(holiday_locale_df['date'].dt.day==1))==False,:] #drop national holiday 01-01
#filter
holiday_locale_df = holiday_locale_df.loc[((holiday_locale_df['date'].dt.month==12)&(holiday_locale_df['date'].dt.day>24))==False,:]
holiday_locale_df = holiday_locale_df.where(holiday_locale_df['transferred']==False).dropna(axis=0,how='all').\
                        loc[:,['date','locale','locale_name']]

for reg_str in ['Local','Regional']:
    holiday_locale_df[reg_str] = list(map(lambda x:1 if x == reg_str else 0,holiday_locale_df['locale']))

store_tmp_df = store_df.rename({'city':'locale_name','state':'locale_name'},axis=1)
store_cmb_df = pd.concat([store_tmp_df.iloc[:,[0,1,3,4]],store_tmp_df.iloc[:,[0,2,3,4]]],axis = 0)
holiday_store_df = pd.merge(holiday_locale_df,store_cmb_df.iloc[:,[0,1]],left_on='locale_name',right_on='locale_name',how='outer')
holiday_store_df.iloc[:,-1] = holiday_store_df.iloc[:,-1].fillna(0).astype(np.int32)
holiday_store_df.dropna(inplace=True,axis=0)
# date_store_key
train_tmp_df = train_df.copy()
train_tmp_df.set_index(['date','store_nbr'],inplace=True)
holiday_store_df.set_index(['date','store_nbr'],inplace=True)
train_tmp_df.loc[:,'id'] = list(range(len(train_tmp_df)))
train_tmp2_df = pd.merge(train_tmp_df,holiday_store_df.iloc[:,2:],left_index=True,right_index=True,how='left')
train_tmp2_df.drop_duplicates(['id'],inplace=True)
#merge national holiday
national_holiday_df = holiday_locale_df.where(holiday_locale_df['locale']=='National').loc[:,['date']].dropna(axis=0).reindex()
national_holiday_df.loc[:,'National'] = 1
train_tmp2_df = pd.merge(train_tmp2_df.reset_index(),national_holiday_df,left_on='date',right_on='date',how='left')
train_tmp2_df.fillna(0,inplace=True)
train_tmp2_df.drop_duplicates(['id'],inplace=True)
train_tmp3_df = train_tmp2_df.copy()
del train_tmp2_df

# days before holiday
date_df = pd.DataFrame(train_df['date'].unique(),columns=['date'])
date_array = date_df.values
national_date_array = national_holiday_df['date'][9:].values
date_array = date_array.repeat(national_date_array.shape[0],axis=1)
national_date_array = national_date_array.reshape(1,-1)
days_gap_array = (date_array-national_date_array).astype('timedelta64[D]').astype(np.int32)
date_pos_gap_array = -1*days_gap_array.copy()
date_pos_gap_array[date_pos_gap_array<0] = 1700
gap_nextholi_array = date_pos_gap_array.min(axis=1)
date_neg_gap_array = -1*days_gap_array.copy()
date_neg_gap_array[date_neg_gap_array>0] = -1700
gap_lastholi_array = date_neg_gap_array.max(axis=1)
holiday_gap_df = date_df.copy()
holiday_gap_df['next_holiday'] = gap_nextholi_array
holiday_gap_df['last_holiday'] = gap_lastholi_array
holiday_gap_df['first_day'] = [1 if date.day_of_year ==1  else 0 for date in date_df['date']]
holiday_gap_df['pre_end'] = [1 if date.day_of_year >=364  else 0 for date in date_df['date']]
#oil data

oil_df = pd.merge(date_df,oil_df,left_on='date',right_on='date',how='left')
oil_df.iloc[:,1].interpolate(limit_direction='forward',inplace=True)
oil_df.iloc[:,1].interpolate(limit_direction='backward',inplace=True)
oil_df.loc[:,'chg_rate'] = np.exp(np.log(oil_df.iloc[:,-1]).diff())-1
oil_df.fillna(0,inplace=True)
oil_df.loc[:,'chg_ma_rate'] =oil_df['chg_rate'].ewm(alpha=0.5,axis=0).mean()
oil_df.loc[:,'oil_ma'] = oil_df['dcoilwtico'].rolling(5,min_periods=1).mean()
oil_df.loc[:,'oil_ma_lag1'] = oil_df.loc[:,'oil_ma'].shift(1).values
oil_df.loc[:,'oil_ma_lag3'] = oil_df.loc[:,'oil_ma'].shift(3).values

#date family:
'''family_stat_df = train_df.pivot_table(index=['date'],columns='family',values='sales',aggfunc = 'sum')
sm.tsa.seasonal_decompose(family_stat_df.iloc[:,7],period=360,model='additive').plot()
plt.show()
plt.close()'''
#family sales rank
train_tmp4_df = train_df.copy()
train_tmp4_df.loc[:,'net_sales'] = train_tmp4_df['sales']-train_tmp4_df['onpromotion']
train_tmp4_df.fillna(0,inplace=True)
train_tmp4_pivot_df = train_tmp4_df.pivot_table(index=['date','store_nbr'],values = 'net_sales',columns = 'family')
train_tmp4_pivot_df.iloc[-16*54:,:] = train_tmp4_pivot_df.iloc[-(32+7)*54:-(16+7)*54,:]
train_tmp4_pivot_ranked_df = train_tmp4_pivot_df.rank(axis=1,method = 'min')
train_tmp4_pivot_ranked_df = train_tmp4_pivot_ranked_df.reset_index()
train_tmp4_pivot_ranked_df.index = list(range(len(train_tmp4_pivot_ranked_df)))
train_tmp4_roll_mean_df = train_tmp4_pivot_ranked_df.iloc[:,1:].groupby('store_nbr').rolling(30,min_periods=1).mean()
train_tmp4_roll_mean_df.iloc[-16*54::,:] = train_tmp4_roll_mean_df.iloc[-(32+7)*54:-(16+7)*54,:]
train_tmp4_roll_mean_df = train_tmp4_roll_mean_df.reset_index()
train_tmp4_roll_mean_df.rename({'level_1':'id'},inplace=True,axis=1)
train_tmp4_roll_mean_df.sort_values('id',inplace=True)
train_tmp4_roll_mean_df.set_index('id',inplace=True)
train_stack_df = train_tmp4_roll_mean_df.iloc[:,1:].stack().to_frame()
train_stack_df = train_stack_df.reset_index()

# std
train_std_pivot_df = train_tmp4_pivot_df.copy().reset_index()
train_std60_pivot_df = train_std_pivot_df.iloc[:,1:].groupby('store_nbr').rolling(60,min_periods=1).std()
train_std60_pivot_df.iloc[-16*54:,:] = train_std60_pivot_df.iloc[-16*2*54:-16*54,:]
train_std60_pivot_df.fillna(0,inplace=True)
train_std60_pivot_df.reset_index(inplace=True)
train_std60_pivot_df.rename({'level_1':'id'},inplace=True,axis=1)
train_std60_pivot_df.sort_values('id',inplace=True)
train_std60_pivot_df.set_index('id',inplace=True)
train_std60_stack_df = train_std60_pivot_df.iloc[:,1:].stack().to_frame()
train_std60_stack_df = train_std60_stack_df.reset_index()
del train_std60_pivot_df
del train_tmp4_roll_mean_df,train_tmp4_df
train_std180_pivot_df = train_std_pivot_df.iloc[:,1:].groupby('store_nbr').rolling(180,min_periods=1).std()
train_std180_pivot_df.iloc[-16*54:,:] = train_std180_pivot_df.iloc[-16*2*54:-16*54,:]
train_std180_pivot_df.fillna(0,inplace=True)
train_std180_pivot_df.reset_index(inplace=True)
train_std180_pivot_df.rename({'level_1':'id'},inplace=True,axis=1)
train_std180_pivot_df.sort_values('id',inplace=True)
train_std180_pivot_df.set_index('id',inplace=True)
train_std180_stack_df = train_std180_pivot_df.iloc[:,1:].stack().to_frame()
train_std180_stack_df = train_std180_stack_df.reset_index()
del train_std180_pivot_df
#dealy sales
def create_date_features(df):
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.weekofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    df['year'] = df['date'].dt.year
    #df["is_wknd"] = df.Date.dt.weekday // 4
    #df['is_month_start'] = df.Date.dt.is_month_start.astype(int)
    #df['is_month_end'] = df.Date.dt.is_month_end.astype(int)
    return df
# Random Noise
def random_noise(dataframe):
    return np.random.normal(scale=2.0, size=(len(dataframe),))

# Lag/Shifted Features
def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store_nbr", "family"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

# Rolling Mean Features
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store_nbr", "family"])['sales']. \
                                                          transform(
            lambda x: x.shift(16).rolling(window=window, min_periods=7, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

# Exponentially Weighted Mean Features
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store_nbr", "family"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe
#sale_delay

#store type and cluster
train_store_df = pd.merge(train_df,store_df.iloc[:,[0,3,4]],left_on='store_nbr',right_on='store_nbr',how='left')
#cluster
train_cluster_df = train_store_df.pivot_table(index=['date','family'],columns='cluster',values='sales',aggfunc='mean')
train_cluster_df = train_cluster_df.rank(axis=1,method='min')
train_cluster_stack_df = train_cluster_df.stack().to_frame()
train_cluster_stack_df.rename({0:'cluster_rank'},axis=1,inplace=True)
train_store_cluster_df = train_store_df.copy()
train_store_cluster_df.set_index(['date','family','cluster'],inplace=True)
train_store_cluster_df = pd.merge(train_store_cluster_df,train_cluster_stack_df,left_index=True,right_index=True,how='left')
train_store_cluster_df.iloc[54*33*(-16):,5:] = train_store_cluster_df.iloc[54*33*(-32-7):54*33*(-16-7),5:].values

#type
train_type_df = train_store_df.pivot_table(index=['date','family'],columns='type',values='sales',aggfunc='mean')
train_type_df = train_type_df.rank(axis=1,method='min')
train_type_stack_df = train_type_df.stack().to_frame()
train_type_stack_df.rename({0:'type_rank'},axis=1,inplace=True)
train_store_type_df = train_store_df.copy()
train_store_type_df.set_index(['date','family','type'],inplace=True)
train_store_type_df = pd.merge(train_store_type_df,train_type_df,left_index=True,right_index=True,how='left')
train_store_type_df.iloc[54*33*(-16):,5:] = train_store_type_df.iloc[54*33*(-32-7):54*33*(-16-7),5:].values
del train_type_stack_df,train_type_df,train_cluster_stack_df,train_cluster_df

def is_payday(day:dt.datetime,lag_int = 0)->int:
    if day.day-lag_int==15:
        return 1
    next_day_dt = day+dt.timedelta(days=1-lag_int)
    if next_day_dt.month-day.month==1:
        return 1
    return 0

#combine data
train_cmb_df = train_df.copy()
train_cmb_df.loc[:,['Local','Regional','National']] = train_tmp3_df.iloc[:,[6,7,8]].values
train_cmb_df.loc[:,['cluster_rank']] = train_store_cluster_df.iloc[:,-1].values
train_cmb_df.loc[:,['type_rank']] = train_store_type_df.iloc[:,-1].values
train_cmb_df = pd.merge(train_cmb_df,oil_df.iloc[:,[0,3,4,5,6]],how='left',left_on='date',right_on='date')
train_cmb_df.loc[:,'rank'] = train_stack_df.values[:,-1]
#train_cmb_df.loc[:,'std60'] = train_std60_stack_df.values[:,-1]
train_cmb_df.loc[:,'std180'] = train_std180_stack_df.values[:,-1]
train_cmb_df.loc[:,'weekday'] = list(map(lambda x: 0 if x.isoweekday() >5 else 1,train_cmb_df['date'].tolist()))
#train_cmb_df.loc[:,'payday'] =list(map(lambda x: is_payday(x),train_cmb_df['date'].tolist()))
train_cmb_df.loc[:,'paydaylag1'] = list(map(lambda x: is_payday(x,1),train_cmb_df['date'].tolist()))
train_cmb_df.loc[:,'after_earthquake'] = 0
earth_loc_int = np.where(train_cmb_df['date']=='2016-4-16')[0][0]
train_cmb_df.iloc[earth_loc_int:earth_loc_int+33*54*7*8,-1] = 1

#time trend
'''train_cmb_df.loc[:,'trend'] = (pd.to_datetime(train_cmb_df.loc[:,'date'])-\
                               dt.datetime.strptime('20121231','%Y%m%d')).month'''

#delay sales
train_cmb_df = create_date_features(train_cmb_df)
train_cmb_df = lag_features(train_cmb_df, lags = [16,17,18,19,20,21,22,30,31,90,180,365])
train_cmb_df = roll_mean_features(train_cmb_df,[16,17,18,30])

alphas = [0.95,0.9, 0.8, 0.5]
lags =[1, 7,30]
train_cmb_df = ewm_features(train_cmb_df, alphas, lags)

train_cmb_df['sales'] = np.log1p(train_cmb_df["sales"].values)

'''train_cmb_df.loc[:,'delay1*paydaylag1'] = train_cmb_df.loc[:,'sales_delay1']*train_cmb_df.loc[:,'paydaylag1']
train_cmb_df.loc[:,'delay1*weekday'] = train_cmb_df.loc[:,'sales_delay1']*train_cmb_df.loc[:,'weekday']
train_cmb_df.loc[:,'delay1*rank*weekday'] = train_cmb_df.loc[:,'sales_delay1']*train_cmb_df.loc[:,'rank']*train_cmb_df.loc[:,'weekday']'''
train_cmb_df.loc[:,'rank*Local'] = train_cmb_df.loc[:,'rank']*train_cmb_df.loc[:,'Local']
train_cmb_df.loc[:,'rank*Regional'] = train_cmb_df.loc[:,'rank']*train_cmb_df.loc[:,'Regional']
train_cmb_df.loc[:,'rank*National'] = train_cmb_df.loc[:,'rank']*train_cmb_df.loc[:,'National']
#train_cmb_df['promotion_rate'] = train_cmb_df['onpromotion']/train_cmb_df['sales_delay1']
#train_cmb_df['promotion_rate'] = train_cmb_df['promotion_rate'].replace([np.inf,np.nan],0)
#split data
train_cmb_df = train_cmb_df.iloc[54*33*365:,:].reset_index()
train_cmb_df = pd.merge(train_cmb_df,holiday_gap_df,left_on='date',right_on='date',how='left')
train_cmb_df = train_cmb_df.set_index(['date','store_nbr','family'])

test_sp_df = train_cmb_df.iloc[-28512:,:]
train_sp_df = train_cmb_df.iloc[:-28512,:]
#ols
model = sm.OLS(train_sp_df['sales'].astype(np.float32),train_sp_df.iloc[:,2:].astype(np.float32))
result = model.fit()
print(result.summary())
train_sp_df.to_pickle('./data/sale_forcast/features_train.pkl')
test_sp_df.to_pickle('./data/sale_forcast/features_test.pkl')
pass



