


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
from skopt import BayesSearchCV
import lightgbm as lgb
import gc
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold

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
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')

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
            gp.to_csv(filename, index=False)
            del gp 

    print(train_df.shape)
    gc.collect()


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
                    gp.to_csv('debug_'+filename, index=False)
                else:
                    gp.to_csv(filename,index=False)
            
        del gp
        gc.collect()   

    print("variables and data type: ")
    train_df.info()
    target = 'is_attributed'



    ### 有问题 需要解决
    predictors.extend(initial_cols)
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    # for i in range(0,naddfeat):
    #     predictors.append('X'+str(i))
        
    
    y = train_df[target]
    X = train_df[predictors]

    bayes_cv_tuner = BayesSearchCV(
        estimator = lgb.LGBMRegressor(
            objective='binary',
            metric='auc',
            n_jobs=1,
            verbose=0
        ),
        search_spaces = {
            # 'learning_rate': (0.01, 1.0, 'log-uniform'),
            'num_leaves': (7, 4095),      
            'max_depth': (2, 63),
            'min_child_samples': (6, 200),
            # 'max_bin': (100, 1000),
            'subsample': (0.4, 1.0, 'uniform'),
            'subsample_freq': (0, 10),
            'colsample_bytree': (0.4, 1.0, 'uniform'),
            'min_child_weight': (0, 10),
            'subsample_for_bin': (100000, 500000),
            'reg_lambda': (1e-9, 1000, 'log-uniform'),
            'reg_alpha': (1e-9, 1.0, 'log-uniform'),
            'scale_pos_weight': (200, 500),
            'min_child_weight': (0.01, 40000),
            'n_estimators': (50, 2000),
        },    
        scoring = 'roc_auc',
        cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=42
        ),
        n_jobs = 3,
        n_iter = ITERATIONS,   
        verbose = 0,
        refit = True,
        random_state = 42
    )

    def status_print(optim_result):
        """Status callback durring bayesian hyperparameter search"""
        # Get all the models tested so far in DataFrame format
        all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
        
        # Get current parameters and the best parameters    
        best_params = pd.Series(bayes_cv_tuner.best_params_)
        print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
            len(all_models),
            np.round(bayes_cv_tuner.best_score_, 4),
            bayes_cv_tuner.best_params_
        ))
        
        # Save all model results
        clf_name = bayes_cv_tuner.estimator.__class__.__name__
        all_models.to_csv(clf_name+"_cv_results.csv")

    # Fit the model
    result = bayes_cv_tuner.fit(X.values, y.values, callback=status_print)     




#===============================================================================================================================================================


start = time.time()
  
debug, gcloud = [0, 1]
# SETTINGS - CHANGE THESE TO GET SOMETHING MEANINGFUL
ITERATIONS = 200 # 1000



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

    val_size = 2500000
    # nchunk = 40000000
    nchunk = 90000000 #from 78000000
    frm = nrows - nchunk
    test_nrows = 18790470
    
to = frm + nchunk

combination = [
    (
        [
            [['ip', 'channel'], 4],
            # [['app', 'channel'], 4],
            [['ip', 'device', 'os', 'app'], 4],
            [['ip', 'app', 'device', 'os'], 'NC']
        ], 
        [
            # ['app', 'os'],
            # ['app', 'channel'],
            ['channel']
        ],
        'Bayesian_opt_40M_300',
        ['ip', 'app','os', 'channel', 'device', 'hour'],
    ),
                                         

]







for group, rategroups, fileno, initial_cols in combination:
    DO(frm,to,test_nrows, group, rategroups, fileno, initial_cols)
end = time.time()
duration = end - start
print('Time consumed is: {} hour'.format(duration/3600.))