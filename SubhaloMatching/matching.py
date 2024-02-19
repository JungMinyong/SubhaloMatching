#Match the subhalos in the high- and low- res. simulations,
# or Hydro and DMO simulations

import sys
sys.path.append("/data1/wispedia/GNN/")

import time
import h5py
import numpy as np
from sklearn.neighbors import KDTree
import pickle
import os

import illustris_python as il
import random
from utils import making_buffer, load_progenitors
import pickle

import torch
import torch.nn as nn

from utils import making_buffer, validation, test_valid, datasetIdx_lst_total, load_optimizer, load_progenitors
from cfg import get_cfg
from model import CustomDatasetMatching, custom_collate_matching, Prediction
from torch.utils.data import Dataset, DataLoader, TensorDataset


cfg = get_cfg()

is50 = cfg.tng50
batch_size = cfg.batch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print('num_workers:',cfg.num_workers)


if is50:
    Boxsize = 35000
    savename='50'
else:
    Boxsize = 75000
    savename='100'
    

isDMO = False
print('isDMO:', isDMO)
print('is50:',is50)
print('BoxSize:', Boxsize)

    
if isDMO:
    modelPath = 'models_valid' #Directory to the saved model
    path = 'models_valid' #Directory that the matching catalog will be stored

else:
    modelPath = 'models_training'
    path = 'models_training'

    
for match12 in [True, False]:
    
    #match12=True: Match subhalos in TNG-1 to those in TNG-2
    #match21=False: subhalos in TNG-2 to TNG-1
    print('match12:',match12)
    
    tree1, tree2, ID1, ID2 = load_progenitors(is50=is50, isDMO=isDMO)
    
    name12 = '12'
    factor_dmo = 0.83 #Omega_dm / Omega_matter
    if not match12: #reverse tree1 and tree2
        name12 = '21'
        temp = tree1
        tree1 = tree2
        tree2 = temp
        
        temp = ID1
        ID1 = ID2
        ID2 = temp
        factor_dmo = 1/0.83 #Omega_matter / Omega_dm



        
    buffer = 5000 #5Mpccm/h
    xyzBuffer, indexBuffer = making_buffer(tree2[:,-1,:3], buffer, Boxsize)
    kdtree2 = KDTree(xyzBuffer, leaf_size=400)

    np.seterr(invalid='ignore')
    
    if is50:
        ab_lst = [2e-2, 5e-2, 1e-1, 1, 1e1, np.inf] #Mass bins [1e10 Msun]
    else:
        ab_lst = [2e-1, 5e-1, 1, 3, 1e1, np.inf]


    dtype = torch.float32
    dataset_full = CustomDatasetMatching(tree1,tree2,ID1,ID2, factor_dmo = factor_dmo, L_box=Boxsize)   
    idx_lst = datasetIdx_lst_total(tree1, ab_lst)
    dataset_lst = [torch.utils.data.Subset(dataset_full, idx) for idx in idx_lst]

    dataloader_lst = [DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers,
                                   pin_memory=False, collate_fn=custom_collate_matching) for dataset in dataset_lst]
    model_lst = []
    
    #==== Load the best models in each mass bin =====
    if is50:
        for i in range(1,len(ab_lst)):
            model = Prediction()
            model.load_state_dict(torch.load(modelPath + '/weights_param_TNG50_{}_best.pth'.format(i)))
            model_lst.append(model)
    else:
        for i in range(1,len(ab_lst)):
            model = Prediction()
            model.load_state_dict(torch.load(modelPath + '/weights_param_TNG100_{}_best.pth'.format(i)))
            model_lst.append(model)
    #================================================
    
    
    num_bin = 0
    
    id1_lst = []
    id2_lst = []
    prob_lst = []
    with torch.no_grad():
        for i, dataloader in enumerate(dataloader_lst):
            model = model_lst[i].to(device)
            model.eval()

            num_bin +=1
            softmax = nn.Softmax(dim=1)

            for dists, masses, id2_arr, masses_z0, id1 in dataloader:
                dists = dists.to(device)
                masses = masses.to(device)
                id2_arr = id2_arr.to(device)
                masses_z0 = masses_z0.to(device)

                final_score = model(dists, masses, masses_z0)
                max_indices = torch.argmax(final_score, axis=1).unsqueeze(1)
                max_softmax_scores = torch.gather(softmax(final_score), 1, max_indices).squeeze(1)

                prob = max_softmax_scores.cpu().numpy() 

                id2 = torch.gather(id2_arr, 1, max_indices).squeeze(1).cpu().numpy()

                id1_lst.extend(id1.cpu().numpy())
                id2_lst.extend(id2)
                prob_lst.extend(prob)

    if match12:
        np.save(path+f'/match12_{savename}', [id1_lst,id2_lst,prob_lst])
    else:
        np.save(path+f'/match21_{savename}', [id2_lst,id1_lst,prob_lst])

np.save(path+f'/bins_{savename}', ab_lst)