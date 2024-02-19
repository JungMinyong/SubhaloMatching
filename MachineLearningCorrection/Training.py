#Training step (including hyperparameter tuning)

import sys
import h5py
import numpy as np
from sklearn.neighbors import KDTree
import illustris_python as il
import random
import time

import os
import pickle
import utils
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

import lightgbm as lgb
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument('-c', '--clf', action='store_true')    
parser.add_argument('-f', '--feat')
parser.add_argument('-s', '--snap')
clf = parser.parse_args().clf
print('is_clf:', clf)

import multiprocessing
n_jobs = multiprocessing.cpu_count()

max_evals = 200
early_stopping_rounds = 40
print(f'n_jobs: {n_jobs}')
y_index = int(parser.parse_args().feat) #0:dm, 1:star, 2:gas, 3:gasmetal
N_snap = int(parser.parse_args().snap)
print(f'N_snap is: {N_snap}') #snap=0 for the full snapshot


if y_index == 3:
    featname = 'metal'
    
elif y_index == 1:
    featname = 'star'
    
elif y_index == 2:
    featname = 'gas'
    
else:
    raise "Invalid y_index"

    
if clf:
    file_name = "new_models/lgbm22_" + featname + f"_clf_{N_snap}snap_newCatalogs.pkl"
else:
    file_name = "new_models/lgbm22_" + featname + f"_reg_{N_snap}snap_newCatalogs.pkl"

container = np.load('.../train_test_4y_tng100-2-Sublink_22feats_newCatalogs2.npz')

is_log = True
print(y_index, is_log, 'y index')
print('Feature:', featname)

X_train = container['X_train']
#X_test = container['X_test']
y_train = container['y_train'][:,y_index]
#y_test = container['y_test'][:,y_index]

N_feat = X_train.shape[-1]

if clf:
    y_train = (y_train>0).astype(int)
    
    print(np.sum(y_train==0), np.sum(y_train != 0))
    eval_metric = 'auc' #rmse
    boosting_type_lst = ['goss','dart'] 
    max_depth_lst = [-1,11,12]
    search_space = {
            'num_leaves': hp.quniform('num_leaves', 20, 160, 1),
            'learning_rate': hp.loguniform('learning_rate', -4, -1),
            'subsample_for_bin': hp.quniform('subsample_for_bin', 25000, 300000, 20000),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1), #alias "subsample"
            'lambda_l1': hp.loguniform('lambda_l1', -16, 2),
            'lambda_l2': hp.loguniform('lambda_l2', -16, 2),
            'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
            'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
            'boosting_type': hp.choice('boosting_type', boosting_type_lst),
            'max_depth': hp.choice('max_depth', max_depth_lst),
    }

else:
    msk_nonzero = y_train>0
    X_train = X_train[msk_nonzero]
    y_train = y_train[msk_nonzero]
    
    if is_log:
        y_train = np.log10(y_train)

    eval_metric = 'rmse'
    boosting_type_lst = ['gbdt', 'dart']
    max_depth_lst = [-1, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    search_space = {
            'num_leaves': hp.quniform('num_leaves', 20, 160, 1),
            'learning_rate': hp.loguniform('learning_rate', -4, -1),
            'subsample_for_bin': hp.quniform('subsample_for_bin', 25000, 300000, 20000),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
            'lambda_l1': hp.loguniform('lambda_l1', -16, 2),
            'lambda_l2': hp.loguniform('lambda_l2', -16, 2),
            'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
            'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
            'boosting_type': hp.choice('boosting_type', boosting_type_lst),
            'max_depth': hp.choice('max_depth', max_depth_lst),    
    }

# labeling
features = np.zeros((100,23), dtype=object)

name_lst=['SubhaloMassType1','SubhaloMassType4','SubhaloMassType0','SubhaloBHMass',
          'SubhaloVmax','SubhaloVelDisp','SubhaloSpin','VelSpinAngle',
          'SubhaloVel','SubhaloStarMetallicity','SubhaloGasMetallicity',
          'SubhaloGasMetallicityMaxRad','SubhaloHalfmassRadType1',
          'SubhaloHalfmassRadType4','SubhaloHalfmassRadType0','SubhaloSFR','GroupMassType1','GroupMassType4',
          'GroupMassType0','Group_R_TopHat200','GroupGasMetallicity','HostDistance','MainBranchRatio']

for j in range(23):
    name = name_lst[j]
    features[:,j] = np.array([name+'_'+str(i) for i in range(100)])

# only take snapshot
X_train = X_train.reshape(-1,100,23)[:, N_snap:,:].reshape(-1, (100-N_snap)*23)
features = features[N_snap:, :]

X_train = pd.DataFrame(X_train, columns=features.flatten())


print('fin preprocess')
import time
t= time.time()

if clf:
    LGBMmodel = LGBMClassifier
    objective = 'binary'
else:
    LGBMmodel = LGBMRegressor
    objective = 'regression'
    
def objective_func(search_space):
    lgbm_clf = LGBMmodel(
        objective = objective,
        n_estimators=1500,
        num_leaves=int(search_space['num_leaves']),    
        min_child_weight=int(search_space['min_child_weight']),
        learning_rate=search_space['learning_rate'], 
        subsample_for_bin= int(search_space['subsample_for_bin']),
        feature_fraction = search_space['feature_fraction'],
        bagging_fraction = search_space['bagging_fraction'],
        min_data_in_leaf = int(search_space['min_data_in_leaf']),
        lambda_l1 = search_space['lambda_l1'],
        lambda_l2 = search_space['lambda_l2'],
        max_depth=search_space['max_depth'],
        boosting_type=search_space['boosting_type'],
        verbose = -1,
        n_jobs = n_jobs
      )

    score_list = []

    ## 3-fold method
    kfold = KFold(n_splits=3)

    for train_index, val_index in kfold.split(X_train):
        X_train_kf, y_train_kf = X_train.iloc[train_index], y_train[train_index]
        X_val_kf, y_val_kf = X_train.iloc[val_index], y_train[val_index]
        
        lgbm_clf.fit(
            X_train_kf, y_train_kf,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric=eval_metric, #'auc', #rmse
            eval_set=[(X_train_kf, y_train_kf), (X_val_kf, y_val_kf)],
            verbose= -1
        )
        
        if clf:        
            score = - roc_auc_score(
                y_val_kf,
                lgbm_clf.predict_proba(X_val_kf)[:, 1])
        else:
            score = mean_squared_error(
                y_val_kf,
                lgbm_clf.predict(X_val_kf))

        score_list.append(score)

    return np.sum(score_list)

trials = Trials()

best = fmin(
    fn=objective_func,
    space=search_space,
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=trials
)


print((time.time()-t) / 3600, 'hours')

best['num_leaves'] = int(best['num_leaves'])
best['min_child_weight'] = int(best['min_child_weight'])
best['min_data_in_leaf'] = int(best['min_data_in_leaf'])
best['subsample_for_bin'] = int(best['subsample_for_bin'])
best['boosting_type'] = boosting_type_lst[int(best['boosting_type'])]
best['max_depth'] = max_depth_lst[int(best['max_depth'])]

print(best)

X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = LGBMmodel(
        objective = objective,
        n_estimators=1500,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric=eval_metric, #'auc' or 'rmse'
        **best)

model.fit(
        X_train1,
        y_train1,
        eval_set=[(X_train1, y_train1), (X_train2, y_train2)],
        verbose=20)

print(file_name)
pickle.dump(model, open(file_name, "wb"))

print('end')
