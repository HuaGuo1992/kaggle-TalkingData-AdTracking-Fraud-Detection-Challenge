


#  base_case + 寻找最优参数
#



"""
A non-blending lightGBM model that incorporates portions and ideas from various public kernels
This kernel gives LB: 0.977 when the parameter 'debug' below is set to 0 but this implementation requires a machine with ~32 GB of memory
"""

import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt
import os


def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.2,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'metric':metrics
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

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])

    return (bst1,bst1.best_iteration)



# ====================================================================================================================================================================
#===============================================Main Function==============================================================================================


def DO(train_frm,train_to, test_nrows, groups, rategroup, fileno, initial_cols=['ip', 'app','device','os', 'channel', 'hour']):
    predictors=[]
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }

    print('loading train data...',frm,to)
    train_df = pd.read_csv(inputpath + "train.csv", parse_dates=['click_time'], skiprows=range(1,train_frm), nrows=train_to-train_frm, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

    print('loading test data...')
    # if debug:
    #     test_df = pd.read_csv(inputpath+"test.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    # else:
    test_df = pd.read_csv(inputpath+"test.csv", nrows=test_nrows, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    test_df['hour'] = pd.to_datetime(test_df.click_time).dt.hour.astype('uint8')


    # Find frequency of is_attributed for each unique value in column
    freqs = {}
    for cols in rategroups:
        def rate_calculation(x):
            """Calculate the attributed rate. Scale by confidence"""
            rate = x.sum() / float(x.count())
            conf = np.min([1, np.log(x.count()) / log_group])
            return rate * conf        
        
        # New feature name
        new_feature = '_'.join(cols)+'_confRate'  
        predictors.append(new_feature)
        filename = new_feature + '.csv'
        if os.path.exists(filename):
            gp=pd.read_csv(filename)
            train_df = train_df.merge(gp, on=cols, how='left') 
            test_df = test_df.merge(gp, on=cols, how='left') 
        else:
            # Perform the groupby
            group_object = train_df.groupby(cols)
            
            # Group sizes    
            group_sizes = group_object.size()
            log_group = np.log(100000) # 1000 views -> 60% confidence, 100 views -> 40% confidence 
            print(">> Calculating confidence-weighted rate for: {}.\n   Saving to: {}. Group Max /Mean / Median / Min: {} / {} / {} / {}".format(
                cols, new_feature, 
                group_sizes.max(), 
                np.round(group_sizes.mean(), 2),
                np.round(group_sizes.median(), 2),
                group_sizes.min()
            ))
            
            # Aggregation function
            
            gp = group_object['is_attributed'].apply(rate_calculation).reset_index().rename( index=str, columns={'is_attributed': new_feature})[cols + [new_feature]]
            # Perform the merge
            train_df = train_df.merge(gp, on=cols, how='left')
            test_df = test_df.merge(gp, on=cols, how='left')
            gp.to_csv(filename, index=False)
            del gp 

    print(train_df.shape)
    gc.collect()

    print('shape of train: ', train_df.shape)
    print('shape of test: ', test_df.shape)
    # print('train.head: ')
    # print(train_df.head())
    # print('test head: ')
    # print(test_df.head())
    len_train = len(train_df)
    train_df=train_df.append(test_df)
    # train_df = pd.concat([train_df, test_df], 0)

    del test_df
    gc.collect()

    # def extract_feature(df, col):
    #     filename = col + '.csv'
    #     if os.path.exists(filename):
    #         print('loading from {} file...'.format(filename))
    #         dp = pd.read_csv(filename)
    #         df[col] = dp.values 
    #         del dp
    #     else:
    #         df[col] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')  
    #         df[col].to_csv(filename)         
    # print('Preprocessing click_time...')
    # train_df['click_time'] = (train_df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)


    # train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    # extract_feature(train_df)
    # extract_feature(test_df)
    
    gc.collect()
    # naddfeat=9
    # for i in range(0,naddfeat):
    #     if i==0: selcols=['ip', 'channel']; QQ=4;
    #     if i==1: selcols=['ip', 'device', 'os', 'app']; QQ=5;
    #     if i==2: selcols=['ip', 'day', 'hour']; QQ=4;
    #     if i==3: selcols=['ip', 'app']; QQ=4;
    #     if i==4: selcols=['ip', 'app', 'os']; QQ=4;
    #     if i==5: selcols=['ip', 'device']; QQ=4;
    #     if i==6: selcols=['app', 'channel']; QQ=4;
    #     if i==7: selcols=['ip', 'os']; QQ=5;
    #     if i==8: selcols=['ip', 'device', 'os', 'app']; QQ=4;
    

    # tpye: 
    # 4: nunique 不同selctor 所对应unique value 的数量
    # 5: cumcont
    

    

    for i, item in enumerate(groups):
        selcols = item[0]
        QQ = item[-1]
        print('selcols',selcols,'QQ',QQ)
        
        colname = '_'.join(selcols) + '_' + str(QQ)
        predictors.append(colname)
        filename= colname + '.csv'
        
        if os.path.exists(filename):
            if QQ==5: 
                gp=pd.read_csv(filename,header=None)
                train_df[colname]=gp 
            else: 
                gp=pd.read_csv(filename)
                train_df = train_df.merge(gp, on=selcols[0:-1], how='left')
        else:
            if QQ==0:
                gp = train_df[selcols].groupby(by=selcols[0:-1])[selcols[-1]].count().reset_index().\
                    rename(index=str, columns={selcols[-1]: colname})
                train_df = train_df.merge(gp, on=selcols[0:-1], how='left')
            if QQ==1:
                gp = train_df[selcols].groupby(by=selcols[0:-1])[selcols[-1]].mean().reset_index().\
                    rename(index=str, columns={selcols[-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:-1], how='left')
            if QQ==2:
                gp = train_df[selcols].groupby(by=selcols[0:-1])[selcols[-1]].var().reset_index().\
                    rename(index=str, columns={selcols[-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:-1], how='left')
            if QQ==3:
                gp = train_df[selcols].groupby(by=selcols[0:-1])[selcols[-1]].skew().reset_index().\
                    rename(index=str, columns={selcols[-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:-1], how='left')
            if QQ==4:
                gp = train_df[selcols].groupby(by=selcols[0:-1])[selcols[-1]].nunique().reset_index().\
                    rename(index=str, columns={selcols[-1]: colname})
                train_df = train_df.merge(gp, on=selcols[0:-1], how='left')
            if QQ==5:
                gp = train_df[selcols].groupby(by=selcols[0:-1])[selcols[-1]].cumcount()
                train_df[colname]=gp.values
            if  QQ == 6:
                gp = train_df[selcols].groupby(by=selcols[0:-1])[selcols[-1]].var().reset_index().rename(index=str, columns={selcols[-1]: colname})
                train_df = train_df.merge(gp, on=selcols[0:-1], how='left')
            if QQ == 7:
                gp = train_df[selcols].groupby(by=selcols[0:-1])[selcols[-1]].mean().reset_index().rename(index=str, columns={selcols[-1]: colname})
                train_df = train_df.merge(gp, on=selcols[0:-1], how='left')     
            if QQ == 'NC':
                train_df['click_time'] = (train_df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)                
                gp = (train_df.groupby(selcols).click_time.shift(-1) - train_df.click_time).astype(np.float32)                 
                train_df[colname] = gp
            # if not debug:
            if QQ != 'NC':
                if debug:
                    gp.to_csv('test'+filename, index=False)
                else:
                    gp.to_csv(filename,index=False)
            
        del gp
        gc.collect()   



    # train_df['click_time'] = (train_df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
    # train_df['NC'] = (train_df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) - train_df.click_time).astype(np.float32)




    # print('doing nextClick')
    
    
    # new_feature = 'nextClick'
    # filename='nextClick_%d_%d.csv'%(frm,to)

    # if os.path.exists(filename):
    #     print('loading from save file')
    #     QQ=pd.read_csv(filename).values
    # else:
    #     # D=2**26
    #     # train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) \
    #     #     + "_" + train_df['os'].astype(str)).apply(hash) % D
    #     # click_buffer= np.full(D, 3000000000, dtype=np.uint32)

    #     # train_df['epochtime']= train_df['click_time'].astype(np.int64) // 10 ** 9
    #     # next_clicks= []
    #     # for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
    #     #     next_clicks.append(click_buffer[category]-t)
    #     #     click_buffer[category]= t
    #     # del(click_buffer)
    #     # QQ= list(reversed(next_clicks))
    #     # del train_df['category']
    #     # del train_df['epochtime']
    #     train_df['click_time'] = (train_df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
    #     train_df['nextClick'] = (train_df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) - train_df.click_time).astype(np.float32)        

    #     if not debug:
    #         print('saving')
    #         pd.DataFrame(QQ).to_csv(filename,index=False)

    # train_df[new_feature] = QQ
    # predictors.append(new_feature)

    # train_df[new_feature+'_shift'] = pd.DataFrame(QQ).shift(+1).values
    # predictors.append(new_feature+'_shift')
    
    # del QQ
    # gc.collect()
    # predictors.extend(['nextClick', 'category', 'epochtime', 'nextClick_shift'])







    # print('grouping by ip-day-hour combination...')
    # gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
    # train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
    # del gp
    # gc.collect()

    # print('grouping by ip-app combination...')
    # gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    # train_df = train_df.merge(gp, on=['ip','app'], how='left')
    # del gp
    # gc.collect()

    # print('grouping by ip-app-os combination...')
    # gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    # train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    # del gp
    # gc.collect()

    # Adding features with var and mean hour (inspired from nuhsikander's script)
    # print('grouping by : ip_day_chl_var_hour')
    # gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
    # train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
    # del gp
    # gc.collect()
    # predictors.append()

    # print('grouping by : ip_app_os_var_hour')
    # gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
    # train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    # del gp
    # gc.collect()

    # print('grouping by : ip_app_channel_var_day')
    # gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
    # train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
    # del gp
    # gc.collect()

    # print('grouping by : ip_app_chl_mean_hour')
    # gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
    # print("merging...")
    # train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
    # del gp
    # gc.collect()

    print("variables and data type: ")
    train_df.info()

    # train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
    # train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    # train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

    target = 'is_attributed'



    ### 有问题 需要解决
    predictors.extend(initial_cols)
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    # for i in range(0,naddfeat):
    #     predictors.append('X'+str(i))
        
    print('predictors',predictors)

    test_df = train_df[len_train:]
    val_df = train_df[(len_train-val_size):len_train]
    train_df = train_df[:(len_train-val_size)]

    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    print("test size : ", len(test_df))

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    gc.collect()

    print("Training...")
    start_time = time.time()

    params = {
        'learning_rate': 0.08, #0.2,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
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
                            objective='binary', 
                            metrics='auc',
                            early_stopping_rounds=30, 
                            verbose_eval=True, 
                            num_boost_round=1000, 
                            categorical_features=categorical)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()
    
    print('Plot feature importances...')
    fig = plt.figure(figsize=(20, 20))
    ax = lgb.plot_importance(bst, max_num_features=100, figsize=(20, 15))
    # plt.show()
    
    plt.savefig(fileno+'_importance.png')

    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)
    # if not debug:
    print("writing...")
    sub.to_csv('sub_{}.csv.gz'.format(str(fileno)),index=False,compression='gzip')
    print("done...")
    return sub


#===============================================================================================================================================================

    


# groups = [
#     # [
# #         [['ip', 'app', 'device', 'os'], 'NC']
# #         ],
# # [
# #     [['ip', 'app', 'device', 'os'], 'NC'], [['device', 'os'], 'NC']
# # ],
# # [
# #     [['ip', 'device', 'os', 'app'], 4],
# #     [['ip', 'app', 'device', 'os'], 'NC'],
# #     [['device', 'os'], 'NC']
# # ],
# # [
# #     [['ip', 'channel'], 4],
# #     [['ip', 'device', 'os', 'app'], 4],
# #     [['ip', 'app', 'device', 'os'], 'NC'],
# #     [['device', 'os'], 'NC']
# # ],
# [[['ip', 'channel'], 4],
#     [['app', 'channel'], 4],
#     [['ip', 'device', 'os', 'app'], 4],
#     [['ip', 'app', 'device', 'os'], 'NC']] 

# ]




# rategroups = [        
#     # V1 Features #
#     ###############
#     # ['ip'], ['app'], ['device'], ['os'], ['channel'],
    
#     # # V2 Features #
#     # ###############
#     # ['app', 'channel'],
#     ['app', 'os']
#     # ['app', 'device'],
    
#     # # V3 Features #
#     # ###############
#     # ['channel', 'os'],
#     # ['channel', 'device'],
#     # ['os', 'device'],
    
#     # # Hua's version
#     # ['ip', 'day'],
#     # ['channel', 'day'],
#     # ['os', 'day'],
#     # ['app', 'day'],
#     # ['ip', 'hour'],
#     # ['channel', 'hour'],
#     # ['os', 'hour'],
#     # ['app', 'hour'], 
#     # ['ip', 'channel', 'hour'],
#     # ['ip', 'os', 'hour'] 

# ]

# filenos = [
#     # 'BC1', 
#     # 'BC2', 
#     # 'BC3',
#     #  'BC4', 
#     #  'BC5'
#     'BC7'
#      ]

# # groups = [
# #     # [['ip', 'channel'], 4],
# #     # [['ip', 'device', 'os','app'],5],
# #     # [['ip', 'day', 'hour'], 4],
# #     # [['ip', 'app'], 4],
# #     # [['ip', 'app', 'os'], 4],
# #     # [['ip', 'device'], 4],
# #     # [['app', 'channel'], 4],
# #     # [['ip', 'os'], 5],
# #     # [['ip', 'device', 'os', 'app'], 4],
# #     # [['ip','day','hour','channel'], 0], 
# #     # [['ip', 'app', 'channel'], 0],
# #     # [['ip','app', 'os', 'channel'], 0],
# #     # [['ip','day','channel', 'hour'], 6],
# #     # [['ip','app', 'os', 'hour'], 6],
# #     # [['ip','app', 'channel', 'day'], 6],
# #     # [['ip','app', 'channel','hour'], 7],
# #     # [['ip', 'app', 'device', 'os'], 'NC']
# # ]  



debug, gcloud = [1, 0]



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
    test_nrows = 18790470
    val_size = 25000000
    # nchunk = 40000000
    nchunk = 184000000 #from 78000000
    frm = nrows - nchunk    
    
to = frm + nchunk

combination = [
    # (
    #     [
    #         [['ip', 'channel'], 4],
    #         [['app', 'channel'], 4],
    #         [['ip', 'device', 'os', 'app'], 4],
    #         [['ip', 'app', 'device', 'os'], 'NC']
    #     ], 
    #     [
    #         # ['app', 'os'],
    #         # ['app', 'channel'],
    #         ['channel']
    #     ],
    #     'BC8',
    #     ['ip', 'app','os', 'channel', 'device', 'hour'],
    # ),
    # (
    #     [
    #         [['ip', 'channel'], 4],
    #         [['app', 'channel'], 4],
    #         [['ip', 'device', 'os', 'app'], 4],
    #         [['ip', 'app', 'device', 'os'], 'NC']
    #     ], 
    #     [
    #         # ['app', 'os'],
    #         ['app', 'channel'],
    #         ['channel']
    #     ],
    #     'BC9',
    #     ['ip', 'app','os', 'channel', 'device', 'hour'],
    # ),
    # (
    #     [
    #         [['ip', 'channel'], 4],
    #         [['app', 'channel'], 4],
    #         [['ip', 'device', 'os', 'app'], 4],
    #         [['ip', 'app', 'device', 'os'], 'NC']
    #     ], 
    #     [
    #         # ['app', 'os'],
    #         ['app', 'channel'],
    #         ['channel'],
    #         ['app']
    #     ],
    #     'BC10',
    #     ['ip', 'app','os', 'channel', 'device', 'hour'],
    # ),    
    (
        [
            [['ip', 'channel'], 4],
            # [['app', 'channel'], 4],
            [['ip', 'device', 'os', 'app'], 4],
            [['ip', 'app', 'device', 'os'], 'NC']
        ], 
        [
            ['channel'],
            # ['app', 'channel'],
            # ['app'],           
            # ['app', 'os'] 
            # ['app', 'device']
        ],
        'BC_all_data',
        ['ip', 'app','os', 'channel', 'device', 'hour'],
    ),  
    # (
    #     [ 
    #         [['ip', 'channel'], 4],
    #         [['app', 'channel'], 4],
    #         [['ip', 'device', 'os', 'app'], 4],
    #         [['ip', 'app', 'device', 'os'], 'NC']
    #     ], 
    #     [
    #         # ['app', 'os'],
    #         ['app', 'channel'],
    #         ['channel'],
    #         ['app'],          
    #         # ['app', 'os']  
    #         ['app', 'device'],
    #         ['ip', 'channel']            
    #     ],
    #     'BC12',
    #     ['ip', 'app','os', 'channel', 'device', 'hour'],
    # )  ,
    # (
    #     [ 
    #         [['ip', 'channel'], 4],
    #         [['app', 'channel'], 4],
    #         [['ip', 'device', 'os', 'app'], 4],
    #         [['ip', 'app', 'device', 'os'], 'NC']
    #     ], 
    #     [
    #         # ['app', 'os'],
    #         ['app', 'channel'],
    #         ['channel'],
    #         ['app'],          
    #         # ['app', 'os']  
    #         # ['app', 'device'],
    #         # ['ip', 'channel']            
    #     ],
    #     'BC13',
    #     ['ip', 'app','os', 'channel', 'device', 'hour'],
    # ) ,
    # (
    #     [
    #         [['ip', 'channel'], 4],
    #         [['app', 'channel'], 4],
    #         [['ip', 'device', 'os', 'app'], 4],
    #         [['ip', 'app', 'device', 'os'], 'NC']
    #     ], 
    #     [
    #         # ['app', 'os'],
    #         # ['app', 'channel'],
    #         ['channel'],
    #         ['ip']
    #     ],
    #     'BC14',
    #     ['ip', 'app','os', 'channel', 'device', 'hour'],
    # ),    
    # (
    #     [
    #         [['ip', 'channel'], 4],
    #         [['app', 'channel'], 4],
    #         [['ip', 'device', 'os', 'app'], 4],
    #         [['ip', 'app', 'device', 'os'], 'NC']
    #     ], 
    #     [
    #         # ['app', 'os'],
    #         # ['app', 'channel'],
    #         ['channel'],
    #         ['hour', 'device', 'os']
    #     ],
    #     'BC15',
    #     ['ip', 'app','os', 'channel', 'device', 'hour'],
    # ),  
    # (
    #     [
    #         [['ip', 'channel'], 4],
    #         [['app', 'channel'], 4],
    #         [['ip', 'device', 'os', 'app'], 4],
    #         [['ip', 'app', 'device', 'os'], 'NC']
    #     ], 
    #     [
    #         # ['app', 'os'],
    #         # ['app', 'channel'],
    #         ['channel'],
    #         ['device', 'os', 'channel']
    #     ],
    #     'BC16',
    #     ['ip', 'app','os', 'channel', 'device', 'hour'],
    # ), 
    # (
    #     [
    #         [['ip', 'channel'], 4],
    #         [['app', 'channel'], 4],
    #         [['ip', 'device', 'os', 'app'], 4],
    #         [['ip', 'app', 'device', 'os'], 'NC']
    #     ], 
    #     [
    #         # ['app', 'os'],
    #         # ['app', 'channel'],
    #         ['channel'],
    #         ['ip', 'channel', 'os']
    #     ],
    #     'BC17',
    #     ['ip', 'app','os', 'channel', 'device', 'hour'],
    # ),  
    # (
    #     [
    #         [['ip', 'channel'], 4],
    #         [['app', 'channel'], 4],
    #         [['ip', 'device', 'os', 'app'], 4],
    #         [['ip', 'app', 'device', 'os'], 'NC']
    #     ], 
    #     [
    #         # ['app', 'os'],
    #         # ['app', 'channel'],
    #         ['channel'],
    #         # ['device', 'os', 'channel']
    #         ['device']
    #     ],
    #     'BC18',
    #     ['ip', 'app','os', 'channel', 'device', 'hour'],
    # ),
    # (
    #     [
    #         [['ip', 'channel'], 4],
    #         [['app', 'channel'], 4],
    #         [['ip', 'device', 'os', 'app'], 4],
    #         [['ip', 'app', 'device', 'os'], 'NC']
    #     ], 
    #     [
    #         # ['app', 'os'],
    #         # ['app', 'channel'],
    #         ['channel'],
    #         # ['device', 'os', 'channel']
    #         ['os']
    #     ],
    #     'BC19',
    #     ['ip', 'app','os', 'channel', 'device', 'hour'],
    # ),   
    # (
    #     [
    #         [['ip', 'channel'], 4],
    #         [['app', 'channel'], 4],
    #         [['ip', 'device', 'os', 'app'], 4],
    #         [['ip', 'app', 'device', 'os'], 'NC']
    #     ], 
    #     [
    #         # ['app', 'os'],
    #         # ['app', 'channel'],
    #         ['channel'],
    #         # ['device', 'os', 'channel']
    #         ['hour']
    #     ],
    #     'BC20',
    #     ['ip', 'app','os', 'channel', 'device', 'hour'],
    # ), 
    # (
    #     [
    #         [['ip', 'channel'], 4],
    #         [['app', 'channel'], 4],
    #         [['ip', 'device', 'os', 'app'], 4],
    #         [['ip', 'app', 'device', 'os'], 'NC'],
    #         [['device', 'os'], 0],
            
    #     ], 
    #     [
    #         # ['app', 'os'],
    #         # ['app', 'channel'],
    #         ['channel']
    #     ],
    #     'BC21',
    #     ['ip', 'app','os', 'channel', 'device', 'hour'],
    # ), 
    # (
    #     [
    #         [['ip', 'channel'], 4],
    #         [['app', 'channel'], 4],
    #         [['ip', 'device', 'os', 'app'], 4],
    #         [['ip', 'app', 'device', 'os'], 'NC'],
    #         # [['device', 'os'], 0],
    #         [['device', 'os', 'channel'], 0]
            
    #     ], 
    #     [
    #         # ['app', 'os'],
    #         # ['app', 'channel'],
    #         ['channel']
    #     ],
    #     'BC22',
    #     ['ip', 'app','os', 'channel', 'device', 'hour'],
    # ),   

                                         

]




for group, rategroups, fileno, initial_cols in combination:
    sub=DO(frm,to,test_nrows, group, rategroups, fileno, initial_cols)
