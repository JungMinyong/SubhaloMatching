#Train the model from the DMO-Hydro pair

#python parameter_train-valid.py --bins 123456 --batch 32 --num_worker 48 --eval_every 1 --tng50

import sys
#import illustris_python as il

import time
import h5py
import numpy as np
from sklearn.neighbors import KDTree
import pickle
import os
import random
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import torch.nn as nn

from utils import making_buffer, validation, test_valid, load_optimizer, datasetIdx_lst_validation, datasetIdx_lst_training, load_progenitors
from cfg import get_cfg
from model import CustomDataset, custom_collate, Prediction, custom_loss


cfg = get_cfg()

isTNG50 = cfg.tng50
batch_size = cfg.batch
N_sample = cfg.Nsample
bins_trained = str(cfg.bins)
max_epochs = 4
eval_every = cfg.eval_every #[1,1,1,2,3,3]

#isTest = True: For the validation step, Split the data into Train (60%), Validation (20%), Test (20%).
#isTest = False: For the application, Split the data into Train (80%), Test (20%).
isTest = False

print(f'batch: {batch_size}, N_sample: {N_sample}')
print('WandB:', cfg.wandb)
print('Restart:', cfg.restart)
print('num_workers:',cfg.num_workers)
print('TNG50:', isTNG50)
print('Early stopping - Max epochs:', max_epochs)
print('eval_every:', max_epochs)
print('isTest:', isTest) 



if isTest:
    save_Path = 'model_valid' #Name of the folder in which model will be saved
else:
    save_Path = 'model_training' #Name of the folder in which model will be saved
    
if isTNG50:
    L_box = 35000
    load_Path = '...'
    simname = 'TNG50'
else:
    L_box = 75000
    load_Path = '...'
    simname = 'TNG100'

if cfg.wandb:
    import wandb
    import os
    os.environ["WANDB_MODE"] = "dryrun"
    wandb.init()

print('L_box': L_box)
print(torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print('device': device)

if isTNG50:
    f = h5py.File(".../TNG50-1/subhalo_matching_to_dark.hdf5")
    matching_true = np.array(f["Snapshot_99"]["SubhaloIndexDark_LHaloTree"])


else: #TNG100
    f = h5py.File(".../TNG100-1/subhalo_matching_to_dark.hdf5")
    matching_true = np.array(f["Snapshot_99"]["SubhaloIndexDark_LHaloTree"])


#Mass bin list. It should be the same with the ab_lst in matching.py
if isTNG50:
    ab_lst = [2e-2, 5e-2, 1e-1, 1, 1e1, np.inf] #mass bins for TNG50 [1e10 Msun]
    
else:
    ab_lst = [2e-1, 5e-1, 1, 3, 1e1, np.inf] #mass bins for TNG100 [1e10 Msun]


tree1, tree2, ID1, ID2 = load_progenitors(is50=isTNG50, isDMO=True)

dtype = torch.float32
np.seterr(invalid='ignore')


#======= undersampling (if N_smaple is large enough, no undersampling) ==========
dataset_full = CustomDataset(tree1,tree2,ID1,ID2, factor_dmo = 0.83, L_box=L_box, matching_true=matching_true)
#factor_dmo = Omega_dm / Omega_matter

if isTest:
    index_lst_train, index_lst_valid, index_lst_test = datasetIdx_lst_validation(tree1, N_sample, ab_lst, L_box)
else:
    index_lst_train, index_lst_valid, index_lst_test = datasetIdx_lst_training(tree1, N_sample, ab_lst, L_box)
    
dataset_train_lst = [torch.utils.data.Subset(dataset_full, idx_train) for idx_train in index_lst_train] 
dataset_valid_lst = [torch.utils.data.Subset(dataset_full, idx_valid) for idx_valid in index_lst_valid]
dataset_test_lst = [torch.utils.data.Subset(dataset_full, idx_test) for idx_test in index_lst_test]
#=================================================================================


train_dataloader_lst = [DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers,
                               pin_memory=False, collate_fn=custom_collate) for dataset_train in dataset_train_lst]
valid_dataloader_lst = [DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers,
                             pin_memory=False, collate_fn=custom_collate) for dataset_valid in dataset_valid_lst]
test_dataloader_lst = [DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers,
                             pin_memory=False, collate_fn=custom_collate) for dataset_test in dataset_test_lst]


if cfg.restart:
    model.load_state_dict(torch.load(load_Path))
    
if cfg.wandb:
    wandb.watch(model)
t = time.time()

print('epoch, loss, accuracy, w_mass p1 gamma')

num_bin = 0
for train_dataloader, test_dataloader, valid_dataloader in zip(train_dataloader_lst, test_dataloader_lst, valid_dataloader_lst):
    model = Prediction().to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.adam_l2)
    
    num_bin +=1
    loss_lst_valid = []
    
    print('start bin',num_bin)
    
    if not (str(num_bin) in bins_trained):
        print(num_bin, 'continue')
        continue
        
    for epoch in range(1000):
        loss_lst = []
        
        true = 0
        total = 0
        for dists, masses, y_true, masses_z0, _ in train_dataloader:
            dists = dists.to(device)
            masses = masses.to(device)
            y_true = y_true.to(device)
            masses_z0 = masses_z0.to(device)

            y_pred = model(dists, masses, masses_z0)

            #use custom loss (Cross entropy + additional term)
            l = custom_loss(y_pred, y_true, model,reg_strength=cfg.loss2_l2, large_scale_stride=5, device=device)

            optimizer.zero_grad()
            l.backward()

            optimizer.step()
            
            #===========Check the performance=========
            y_true_max, _ = torch.max(y_true, axis=1)
            y_pred_subset = y_pred[y_true_max==1]
            y_true_subset = torch.argmax(y_true[y_true_max==1],axis=1)
            
            accu = (torch.argmax(y_pred_subset,axis=1)==y_true_subset).detach().cpu().numpy()
            
            true += np.sum(accu)
            total += len(accu)
            loss_lst.extend(l.detach().cpu().numpy().flatten())
            #===========Check the performance=========
            
            
            # Prevent the paramters to have negative values
            for module in model.modules():
                if hasattr(module, 'weight'):
                    w = module.weight.data
                    w = w.clamp(1e-5, None)
                    module.weight.data = w

        print('{}/{}. {:.3e}, {:.5f}'.format(num_bin,epoch,np.mean(loss_lst),true/total))

        #Evaluate the validation set
        if epoch%eval_every==0:
            model.eval()
            trainloss, validloss = validation(model, device, valid_dataloader, np.mean(loss_lst), numloader=5)
            model.train()

            loss_lst_valid.append(validloss)
            
            #Early stopping
            if (len(loss_lst_valid) > max_epochs) and (loss_lst_valid[-1] > max(loss_lst_valid[-max_epochs-1:-1])):
                print('Early Stopping')
                break

            #Save each epoch
            torch.save(model.state_dict(), save_Path+'/weights_param_{}_5bins_{}_{}.pth'.format(simname,num_bin,epoch))
    
    #Save the epoch with the best performance in the validation set
    epoch_best = np.argmin(loss_lst_valid) * eval_every
    path_best = save_Path+'/weights_param_{}_{}_{}.pth'.format(simname,num_bin,epoch_best)
    model.load_state_dict(torch.load(path_best))
    torch.save(model.state_dict(), save_Path+'/weights_param_{}_{}_best.pth'.format(simname,num_bin))
    
np.save(save_Path+f'bins_{simname}', ab_lst)
print('Total Time', (time.time() - t)/3600 )


