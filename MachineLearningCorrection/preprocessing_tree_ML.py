#Preprocessing: Extract 23 features for the training

import sys
import h5py
import numpy as np
import illustris_python as il
import numpy as np
import pandas as pd


def log_mass_with_eps(x, eps):
    x = x * 1e10
    x[x == 0] = eps
    return np.log10(x)

def extend_data(arr):
    assert arr.shape[0]<=100
    assert len(arr.shape) == 2
    
    extend_arr = np.zeros((100-arr.shape[0], arr.shape[1]))
    arr_return = np.concatenate((arr,extend_arr))
    
    return arr_return

BasePath = ".../TNG/TNG100-2/output"

mass = il.groupcat.loadSubhalos(BasePath, 99, fields=["SubhaloMassType"])[:, 1] / 0.6774 * 1e10
my_index = np.arange(len(mass))[mass > 3 * 10**9]

N_total = len(mass)
N_sub = len(my_index)

x_lst = np.zeros( (N_total, 100, 23) ).astype('float32') #10 (shape[2]) features

print(len(my_index))
for i, id in enumerate(my_index):
    if i%10000 == 0:
        print(i, len(my_index))
        
    # ==== load full tree for the MainProgenitorMassFraction feature======
    fields = ['SnapNum','SubhaloMassType']
    tree = il.lhalotree.loadTree(BasePath,99,id,fields=fields,onlyMPB=False)
    df = pd.DataFrame({'snapshot': tree['SnapNum'], 'mass': tree['SubhaloMassType'][:,1]})

    # group by 'snapshot' and sum 'mass'
    df_grouped = df.groupby('snapshot').sum()

    df_grouped_reindexed = df_grouped.reindex(range(100), fill_value=0)
    sum_dm_mass = df_grouped_reindexed.values.reshape(-1) 
    # ==== load full tree for the MainProgenitorMassFraction feature======

    
    # load Main branch
    fields = [
        "SubhaloMassType",
        "SubhaloVelDisp",
        "SubhaloHalfmassRadType",
        "SubhaloPos",
        "SubhaloVmax",
        "SubhaloVelDisp",
        "SubhaloVel",
        "SubhaloSpin",
        'Group_R_TopHat200',
        'SubhaloGasMetallicity',
        'SubhaloBHMass',
        'SubhaloSFR',
        'SubhaloStarMetallicity',
        'SubhaloGasMetallicityMaxRad',
        'GroupPos',
        'GroupBHMass',
        'GroupGasMetallicity',
        'GroupMassType',
        'SnapNum'
    ]

    def log_mass_with_eps(x, eps):
        x = x * 1e10
        x[x == 0] = eps
        return np.log10(x)

    tree = il.sublink.loadTree(BasePath, 99, id, fields=fields, onlyMPB=True)
    x = np.stack(
        (
            log_mass_with_eps(tree["SubhaloMassType"][:, 1], 1),
            log_mass_with_eps(tree["SubhaloMassType"][:, 4], 1),
            log_mass_with_eps(tree["SubhaloMassType"][:, 0], 1),
            log_mass_with_eps(tree["SubhaloBHMass"], 1),
            tree["SubhaloVmax"],
            tree["SubhaloVelDisp"],
            np.linalg.norm(tree["SubhaloSpin"], axis=1),
            np.sum(tree["SubhaloVel"]*tree["SubhaloSpin"],axis=1)/(np.linalg.norm(tree["SubhaloSpin"],axis=1)*np.linalg.norm(tree["SubhaloVel"],axis=1)),
            np.linalg.norm(tree["SubhaloVel"], axis=1),
            tree["SubhaloStarMetallicity"],
            tree["SubhaloGasMetallicity"],
            tree["SubhaloGasMetallicityMaxRad"],
            tree["SubhaloHalfmassRadType"][:, 1],
            tree["SubhaloHalfmassRadType"][:, 4],
            tree["SubhaloHalfmassRadType"][:, 0],
            tree["SubhaloSFR"],
            log_mass_with_eps(tree['GroupMassType'][:,1],1),
            log_mass_with_eps(tree['GroupMassType'][:,4],1),
            log_mass_with_eps(tree['GroupMassType'][:,0],1),
            tree['Group_R_TopHat200'],
            tree['GroupGasMetallicity'],
            np.linalg.norm(tree["SubhaloPos"] - tree["GroupPos"], axis=1),
        )
    ).T
    
    y = np.zeros((100, x.shape[1]+1), dtype='float32')
    y[tree['SnapNum'], :-1] = x
    
    y[:,-1] = y[:,0] - log_mass_with_eps(sum_dm_mass, 1) #Log(mass_main / mass_total), Log(0/0) become 0.    
    x_lst[id] = y.astype('float32')


np.save(f'tng100-2-Sublink_{x.shape[1]}feats_x',x_lst)
