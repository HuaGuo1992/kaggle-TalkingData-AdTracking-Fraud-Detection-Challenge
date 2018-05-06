# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the inputpath + "/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
#import matplotlib.pyplot as plt
import os
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt
import pytz

predictors=[]
def do_next_Click( df,agg_suffix='nextClick', agg_type='float32'):
    
    print(f">> \nExtracting {agg_suffix} time calculation features...\n")
    
    GROUP_BY_NEXT_CLICKS = [
    
    # V1
    # {'groupby': ['ip']},
    # {'groupby': ['ip', 'app']},
    # {'groupby': ['ip', 'channel']},
    # {'groupby': ['ip', 'os']},
    
    # V3
    {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    {'groupby': ['ip', 'os', 'device']},
    {'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        # df[new_feature] = (df[all_features].groupby(spec[
        #     'groupby']).click_time.shift(-1) - df.click_time).dt.seconds.astype(agg_type)
        df[new_feature] = (df[all_features].groupby(spec[
            'groupby']).click_time.shift(-1).subtract(df.click_time)).dt.seconds.astype(agg_type)        
        if new_feature not in predictors:
            predictors.append(new_feature)
        gc.collect()
    return (df)

def do_prev_Click( df,agg_suffix='prevClick', agg_type='float32'):

    print(f">> \nExtracting {agg_suffix} time calculation features...\n")
    
    GROUP_BY_NEXT_CLICKS = [
    
    # V1
    # {'groupby': ['ip']},
    # {'groupby': ['ip', 'app']},
    {'groupby': ['ip', 'channel']},
    {'groupby': ['ip', 'os']},
    
    # V3
    #{'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    #{'groupby': ['ip', 'os', 'device']},
    #{'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        # df[new_feature] = (df.click_time - df[all_features].groupby(spec[
        #         'groupby']).click_time.shift(+1) ).dt.seconds.astype(agg_type)
        df[new_feature] = (df.click_time.subtract(df[all_features].groupby(spec[
                'groupby']).click_time.shift(+1))).dt.seconds.astype(agg_type)        
        
        if new_feature not in predictors:
            predictors.append(new_feature)
        gc.collect()
    return (df)    

def do_count( df, group_cols, agg_type='uint32', show_max=False, show_agg=True ):
    agg_name='{}count'.format('_'.join(group_cols))  
    if show_agg:
        print( "\nAggregating by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    if agg_name not in predictors:
        predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )

def do_countuniq( df, group_cols, counted, agg_type='uint32', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_countuniq'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCounting unqiue ", counted, " by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    if agg_name not in predictors:
        predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )

def do_cumcount( df, group_cols, counted,agg_type='uint16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_cumcount'.format(('_'.join(group_cols)),(counted)) 
    if show_agg:
        print( "\nCumulative count by ", group_cols , '... and saved in', agg_name  )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    if agg_name not in predictors:
        predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )

def do_mean( df, group_cols, counted, agg_type='float16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_mean'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCalculating mean of ", counted, " by ", group_cols , '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    if agg_name not in predictors:
        predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )

def do_var( df, group_cols, counted, agg_type='float16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_var'.format(('_'.join(group_cols)),(counted)) 
    if show_agg:
        print( "\nCalculating variance of ", counted, " by ", group_cols , '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    if agg_name not in predictors:
        predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=50, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.04,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 8,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 4,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0.99,  # L1 regularization term on weights
        'reg_lambda': 0.9,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 1,
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    del dtrain
    del dvalid
    gc.collect()

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[ xgvalid], 
                     valid_names=['valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])

    return (bst1,bst1.best_iteration)

def DO(frm,to,fileno):
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint8',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }

    print('loading train data...',frm,to)
    train_df = pd.read_csv(inputpath + "/train.csv", parse_dates=['click_time'], skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

    print('loading test data...')
    if debug:
        test_df = pd.read_csv(inputpath + "/test_supplement.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv(inputpath + "/test_supplement.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])


    local_tz = pytz.timezone('Asia/Shanghai') 

    def utc_to_local(utc_dt):
        local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
        return local_tz.normalize(local_dt) 


    train_df['click_time'] = train_df['click_time'].apply(utc_to_local)
    test_df['click_time'] = test_df['click_time'].apply(utc_to_local)
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    test_df['hour'] = pd.to_datetime(test_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    test_df['day'] = pd.to_datetime(test_df.click_time).dt.day.astype('uint8')    


    len_train = len(train_df)
    # train_df=train_df.append(test_df)
    
    # del test_df
        
    gc.collect()

    def process_data(data):
        data = do_next_Click( data,agg_suffix='nextClick', agg_type='float32'  ); gc.collect()
        data = do_prev_Click( data,agg_suffix='prevClick', agg_type='float32'  ); gc.collect()  ## Removed temporarily due RAM sortage. 
        
        data = do_countuniq( data, ['ip'], 'channel' ); gc.collect()
        print('data columns', data.columns)
        data = do_countuniq( data, ['ip', 'device', 'os'], 'app'); gc.collect()
        data = do_countuniq( data, ['ip', 'day'], 'hour' ); gc.collect()
        data = do_countuniq( data, ['ip'], 'app'); gc.collect()
        data = do_countuniq( data, ['ip', 'app'], 'os'); gc.collect()
        data = do_countuniq( data, ['ip'], 'device'); gc.collect()
        data = do_countuniq( data, ['app'], 'channel'); gc.collect()
        data = do_cumcount( data, ['ip'], 'os'); gc.collect()
        data = do_cumcount( data, ['ip', 'device', 'os'], 'app'); gc.collect()
        data = do_count( data, ['ip', 'day', 'hour'] ); gc.collect()
        data = do_count( data, ['ip', 'app']); gc.collect()
        data = do_count( data, ['ip', 'app', 'os']); gc.collect()
        data = do_var( data, ['ip', 'app', 'os'], 'hour'); gc.collect()

        del data['day']
        gc.collect()
        return data 
    
    train_df = process_data(train_df)
    print('train_df cols after process', train_df.columns)
    test_df = process_data(test_df)
    print('test_df cols after process', test_df.columns)

    # predictors = list(set(predictors))

    
    
    print('\n\nBefore appending predictors...\n\n',sorted(predictors))
    target = 'is_attributed'
    word= ['app','device','os', 'channel', 'hour']
    for feature in word:
        if feature not in predictors:
            predictors.append(feature)
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    print('\n\nAfter appending predictors...\n\n',sorted(predictors))
    if debug:
        test_df = test_df
    else:
        relation = pd.read_csv(inputpath + 'mapping.csv', usecols=['old_click_id'])

        # test_df = train_df[len_train:]
        test_df = test_df.iloc[relation.old_click_id]
        del relation
    
    val_df = train_df[(len_train-val_size):]
    train_df = train_df[:(len_train-val_size)]

    print("\ntrain size: ", len(train_df))
    print("\nvalid size: ", len(val_df))
    print("\ntest size : ", len(test_df))

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    gc.collect()

    print("Training...")
    start_time = time.time()
    print('predictors', predictors)
    print('train cols', train_df.columns)
    print('test cols', test_df.columns)

    params = {
        'learning_rate': 0.02,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 31,  # 2^max_depth - 1
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 128,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':200 # because training data is extremely unbalanced 
    }
    (bst,best_iteration) = lgb_modelfit_nocv(params, 
                            train_df, 
                            val_df, 
                            predictors, 
                            target, 
                            objective='xentropy', 
                            metrics='auc',
                            early_stopping_rounds=30, 
                            verbose_eval=True, 
                            num_boost_round=2000, 
                            categorical_features=categorical)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()


    print('Plot feature importances...')
    fig = plt.figure(figsize=(20, 20))
    ax = lgb.plot_importance(bst, max_num_features=100, figsize=(20, 15))
    # plt.show()
    
    plt.savefig(str(fileno)+'_importance.png')

    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)
#     if not debug:
#         print("writing...")
    sub.to_csv('sub_it%d.csv'%(fileno),index=False,float_format='%.9f')
    print("done...")
    return sub


FILENO= 34 #To distinguish the output file name3
debug, gcloud = [0, 1]  #Whethere or not in debuging mode    



if gcloud:
    inputpath = '../data/'
else:
    inputpath = '../../'

if debug:
    val_size = 10000
    frm = 0
    nchunk = 100000
    test_nrows = 100000
else:
    nrows = 184903891 - 1  

    # val_size = 2500000
    # nchunk = 40000000
    # nchunk = 90000000 #from 78000000
    # frm = nrows - nchunk
    # test_nrows = 18790470
    # val_size = 25000
    # # nchunk = 40000000
    # nchunk = 100000 #from 78000000
    # # frm = nrows - nchunk     


    test_nrows = 18790470
    val_size = 25000000
    # nchunk = 40000000
    nchunk = 100000000 #from 78000000
    # frm = nrows - nchunk   

    frm=nrows-(nchunk + val_size)

to = frm + nchunk


sub=DO(frm,to,FILENO)
