import sys

sys.path.append("/data1/wispedia/")

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

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score



path1 = ".../TNG/TNG100-1/output"
path2 = ".../TNG/TNG100-2/output"

mass1 = il.groupcat.loadSubhalos(path1, 99, "SubhaloMassType")[:, 1]*1e10
gas1 = il.groupcat.loadSubhalos(path1, 99, "SubhaloMassType")[:, 0]*1e10
star1 = il.groupcat.loadSubhalos(path1, 99, "SubhaloMassType")[:, 4]*1e10
metal_gas1 = il.groupcat.loadSubhalos(path1, 99, "SubhaloGasMetallicity")

#gas1 = il.groupcat.loadSubhalos(path1, 99, "SubhaloMassType")[:, 0]*1e10
#pos1 = il.groupcat.loadSubhalos(path1, 99, "SubhaloPos")

mass2 = il.groupcat.loadSubhalos(path2, 99, "SubhaloMassType")[:, 1]*1e10
#star2 = il.groupcat.loadSubhalos(path2, 99, "SubhaloMassType")[:, 4]*1e10
#gas2 = il.groupcat.loadSubhalos(path2, 99, "SubhaloMassType")[:, 0]*1e10
#pos2 = il.groupcat.loadSubhalos(path2, 99, "SubhaloPos")



def make_matching(matching1, matching2, value1, value2, prob = 0.7):
    matching_final1 = []
    matching_final2 = []
    for key in matching1.keys():
        item1 = matching1[key]
        val1 = value1[key]
        if (item1 in matching2.keys()) & (val1 > prob):
            # print(key, item1, matching2[item1])
            if (key == matching2[item1]) & (value2[item1] > prob):
                matching_final1.append(key)
                matching_final2.append(matching1[key])
            else:
                1# print('warning')
                
    return np.array(matching_final1), np.array(matching_final2)


test = np.load('../models_training/match12_100.npy')
id1 = np.array(test[0], dtype=int)
id2 = np.array(test[1], dtype=int)
prob = test[2]
matching12 = dict(zip(id1, id2))
prob12 = dict(zip(id1, prob))

test = np.load('../models_training/match21_100.npy')
id1 = np.array(test[0], dtype=int)
id2 = np.array(test[1], dtype=int)
prob = test[2]
matching21 = dict(zip(id2, id1))
prob21 = dict(zip(id2, prob))

p_matching = 0.7

id1, id2 = make_matching(matching12, matching21, prob12, prob21, p_matching)






#print(len(dict21), len(dict12), len(id1), len(id2))
print(len(id1), len(id2))


X = np.load('.../tng100-2-Sublink_22feats_x.npy')
N_feat = X.shape[-1]

#X = X[id2]


h = 0.6774
msk = (mass2[id2]/h > 3e9) & (mass1[id1]/h > 3e9)

y = np.stack( (mass1, star1, gas1, metal_gas1), axis=1)
y = y[id1[msk]]
X = X[id2[msk]]

print(X.shape, y.shape)
X = X.reshape(X.shape[0], -1)
print(X.shape, y.shape)

## Split the data into training and testing sets
X_train, X_test, y_train, y_test, id1_train, id1_test, id2_train, id2_test = train_test_split(X, y, id1[msk], id2[msk], test_size=0.2, random_state=42)

np.savez('.../train_test_4y_tng100-2-Sublink_22feats_retry.npz',
         X_train = X_train, X_test = X_test, y_train = y_train, y_test=y_test,
        id1_train = id1_train, id1_test = id1_test, id2_train = id2_train, id2_test = id2_test)

print('fin')
