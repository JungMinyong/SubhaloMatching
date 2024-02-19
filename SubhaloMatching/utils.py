import numpy as np
from sklearn.neighbors import KDTree

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import torch.nn as nn

def making_buffer(xyzData, L, BOXSIZE):
    lst123 = [0, -1, 1]
    data_lst = []
    Index_lst = []
    index = np.arange(len(xyzData))
    
    for i in lst123:
        for j in lst123:
            for k in lst123:
                data_shifted = xyzData + np.array([i, j, k]) * BOXSIZE
                mask = (
                    (data_shifted[:, 0] > -L)
                    & (data_shifted[:, 0] < BOXSIZE + L)
                    & (data_shifted[:, 1] > -L)
                    & (data_shifted[:, 1] < BOXSIZE + L)
                    & (data_shifted[:, 2] > -L)
                    & (data_shifted[:, 2] < BOXSIZE + L)
                )
                data_lst.append(data_shifted[mask])
                Index_lst.append(index[mask])

    return np.concatenate(data_lst), np.concatenate(Index_lst)


def custom_collate(batch): #(2)
    dists_lst, masses_lst, y_lst, masses_z0_lst = [], [], [], []
    
    for (_dists, _masses, _y, _masses_z0) in batch:
        dists_lst.append(_dists)
        masses_lst.append(_masses)
        y_lst.append(_y)
        masses_z0_lst.append(_masses_z0)
        
    dists = pad_sequence(dists_lst, batch_first=True,padding_value=5000)
    masses = pad_sequence(masses_lst, batch_first=True,padding_value=100)
    y = torch.tensor(y_lst)
    masses_z0 = torch.tensor(masses_z0_lst)
    
    return dists, masses, y, masses_z0


def validation(model, device, test_dataloader, trainloss, numloader=4):
    loss = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        loss_lst = []
        true = 0
        total = 0        
        for temp in test_dataloader:
            if numloader == 4:
                dists, masses, y_true, masses_z0 = temp
            elif numloader == 5:
                dists, masses, y_true, masses_z0, _ = temp
            else:
                raise "invalid numloader"
                
            dists = dists.to(device)
            masses = masses.to(device)
            y_true = y_true.to(device)
            masses_z0 = masses_z0.to(device)
            
            y_pred = model(dists, masses, masses_z0)
            l = loss(y_pred, y_true)

            y_true_max, _ = torch.max(y_true, axis=1)
            y_pred_subset = y_pred[y_true_max==1]
            y_true_subset = torch.argmax(y_true[y_true_max==1],axis=1)
            accu = (torch.argmax(y_pred_subset,axis=1)==y_true_subset).detach().cpu().numpy()            
            
            true += np.sum(accu)
            total += len(accu)
            loss_lst.extend(l.detach().cpu().numpy().flatten())
        testloss = np.mean(loss_lst)

    print('test loss =', '{:.6f}'.format(testloss),'test accu. =', '{:.6f}'.format(true/total))

    return trainloss, testloss

def test_valid(model, device, test_dataloader, trainloss):
    loss = nn.CrossEntropyLoss().to(device)
    loss_lst = []
    id1_lst = []
    accu_lst = []
    
    with torch.no_grad():

        true = 0
        total = 0        
        for dists, masses, y_true, masses_z0, id1 in test_dataloader:
            dists = dists.to(device)
            masses = masses.to(device)
            y_true = y_true.to(device)
            masses_z0 = masses_z0.to(device)
            id1 = id1.to(device)
            
            y_pred = model(dists, masses, masses_z0)
            l = loss(y_pred, y_true)

            y_true_max, _ = torch.max(y_true, axis=1)
            y_pred_subset = y_pred[y_true_max==1]
            y_true_subset = torch.argmax(y_true[y_true_max==1],axis=1)
            accu = (torch.argmax(y_pred_subset,axis=1)==y_true_subset).detach().cpu().numpy()            

            true += np.sum(accu)
            total += len(accu)
            loss_lst.extend(l.detach().cpu().numpy().flatten())
            id1_lst.extend(id1.detach().cpu().numpy().flatten())
            accu_lst.extend(accu)
            
        testloss = np.mean(loss_lst)

    print('test loss =', '{:.6f}'.format(testloss),'test accu. =', '{:.6f}'.format(true/total))
    return np.array(id1_lst), np.array(accu_lst)


def test_lst(model, device, test_dataloader_lst, trainloss):
    loss = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        loss_lst = []
        true = 0
        total = 0 
        for test_dataloader in test_dataloader_lst:
            loss_lst_subset = []
            for dists, masses, y_true, masses_z0 in test_dataloader:
                dists = dists.to(device)
                masses = masses.to(device)
                y_true = y_true.to(device)
                masses_z0 = masses_z0.to(device)

                y_pred = model(dists, masses, masses_z0)
                l = loss(y_pred, y_true)

                accu = (torch.argmax(y_pred,axis=1)==y_true).detach().cpu().numpy()
                true += np.sum(accu)
                total += len(accu)
                loss_lst_subset.extend(l.detach().cpu().numpy().flatten())
            print('test loss subset:', '{:.6f}'.format(np.mean(loss_lst_subset)))
            loss_lst.extend(loss_lst_subset)
            
        testloss = np.mean(loss_lst)

    print('test loss =', '{:.6f}'.format(testloss),'test accu. =', '{:.6f}'.format(true/total))
    return trainloss, testloss

def undersampling(tree1, N_sample, ab_lst, L_box):
    index_lst_train = []
    index_lst_test = []

    index_total = np.arange(len(tree1))
    msk2 = tree1[:,-1,2] < 0.8 * L_box
    
    for i in range(len(ab_lst)-1):
        a = ab_lst[i]
        b = ab_lst[i+1]

        msk1 = (tree1[:,-1,3] > a) & (tree1[:,-1,3] < b)
        
        #print(np.sum(msk))
        
        msk_train = msk1 & msk2
        msk_test = msk1 & ~msk2

        N_sample_train = int(0.8*N_sample)#min(np.sum(msk_train), int(0.8*N_sample))
        N_sample_test =  int(0.2*N_sample)#min(np.sum(msk_test), int(0.2*N_sample))
        
        bool_train = N_sample_train > np.sum(msk_train)
        bool_test = N_sample_test > np.sum(msk_test)
            
        print(f'{a} < x <{b}: {sum(msk_train)}, {sum(msk_test)}')
        print(bool_train, bool_test, 'multiple choices')
        #replace = true: A can be selected multiple times
        index_train = np.random.choice(index_total[msk_train], N_sample_train, bool_train)
        index_lst_train.extend(index_train)
        index_test = np.random.choice(index_total[msk_test], N_sample_test, bool_test)
        index_lst_test.extend(index_test)
        
    index_lst_train = np.array(index_lst_train)
    index_lst_test = np.array(index_lst_test)
    return index_lst_train, index_lst_test



def datasetIdx_lst_training(tree1, N_sample, ab_lst, L_box):
    index_lst_train = []
    index_lst_valid = []
    index_lst_test = []
    
    index_total = np.arange(len(tree1))
    msk2 = tree1[:,-1,2] < 0.8 * L_box
    
    for i in range(len(ab_lst)-1):
        a = ab_lst[i]
        b = ab_lst[i+1]

        msk1 = (tree1[:,-1,3] > a) & (tree1[:,-1,3] < b)
        
        #print(np.sum(msk))
        
        msk_train = msk1 & msk2
        msk_test = msk1 & ~msk2

        N_sample_train = min(np.sum(msk_train), int(0.8*N_sample))
        N_sample_test =  min(np.sum(msk_test), int(0.2*N_sample))
            
        print(f'{a} < x <{b}: {sum(msk_train)}, {sum(msk_test)}')
        #replace = true: A can be selected multiple times
        index_train = np.random.choice(index_total[msk_train], N_sample_train, False)
        index_test = np.random.choice(index_total[msk_test], N_sample_test, False)

        index_lst_train.append(index_train)
        index_lst_test.append([0])
        index_lst_valid.append(index_test)
        
    return index_lst_train, index_lst_valid, index_lst_test


def datasetIdx_lst_validation(tree1, N_sample, ab_lst, L_box):
    index_lst_train = []
    index_lst_test = []
    index_lst_valid = []
    
    index_total = np.arange(len(tree1))
    msk2_train = tree1[:,-1,2] < 0.6 * L_box
    msk2_valid = (tree1[:,-1,2] >= 0.6 * L_box) & (tree1[:,-1,2] < 0.8 * L_box)
    msk2_test = tree1[:,-1,2] >= 0.8 * L_box
    
    for i in range(len(ab_lst)-1):
        a = ab_lst[i]
        b = ab_lst[i+1]

        msk1 = (tree1[:,-1,3] > a) & (tree1[:,-1,3] < b)
        
        #print(np.sum(msk))
        
        msk_train = msk1 & msk2_train
        msk_test = msk1 & msk2_test
        msk_valid = msk1 & msk2_valid
        
        N_sample_train = min(np.sum(msk_train), int(0.8*N_sample))
        N_sample_test =  min(np.sum(msk_test), int(0.2*N_sample))
        N_sample_valid =  min(np.sum(msk_valid), int(0.2*N_sample))
            
        print(f'{a} < x <{b}: {sum(msk_train)}, {sum(msk_test)}')
        #replace = true: A can be selected multiple times
        index_train = np.random.choice(index_total[msk_train], N_sample_train, False)
        index_test = np.random.choice(index_total[msk_test], N_sample_test, False)
        index_valid = np.random.choice(index_total[msk_valid], N_sample_valid, False)

        index_lst_train.append(index_train)
        index_lst_test.append(index_test)
        index_lst_valid.append(index_valid)
        
    return index_lst_train, index_lst_valid, index_lst_test


def datasetIdx_lst_total(tree1, ab_lst):
    index_lst = []
    
    index_total = np.arange(len(tree1))
    
    for i in range(len(ab_lst)-1):
        a = ab_lst[i]
        b = ab_lst[i+1]

        msk1 = (tree1[:,-1,3] > a) & (tree1[:,-1,3] < b)
           
        print(f'{a} < x <{b}: {sum(msk1)}')
        index_lst.append(index_total[msk1])
        
    return index_lst


def load_optimizer(i, adam_l2):
    if i==3:
        optimizer = torch.optim.Adam([
            {'params': model.linear1.parameters(), 'lr': 0.01},
            {'params': model.gamma, 'lr': 0.01},
            {'params': model.p1, 'lr': 0.03},
            {'params': model.w_mass, 'lr': 0.03}], weight_decay=adam_l2)
    else:
        optimizer = torch.optim.Adam([
            {'params': model.linear1.parameters(), 'lr': 0.01},
            {'params': model.gamma, 'lr': 0.01},
            {'params': model.p1, 'lr': 0.03},
            {'params': model.w_mass, 'lr': 0.03}], weight_decay=adam_l2)

        
def custom_loss4(y_pred, y_true, model, reg_strength=1e-4, large_scale_stride=5):
    # Calculate the original loss (e.g., mean squared error)
    criterion = nn.CrossEntropyLoss().to(device)
    ce_loss = criterion(y_pred, y_true)

    # Calculate the regularization term for local smoothness
    reg_term_local = 0
    weight_diff_local = torch.abs(param[:, 1:] - param[:, :-1])
    reg_term_local += torch.sum(weight_diff_local)

    # Calculate the regularization term for large-scale smoothness
    reg_term_large_scale = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_diff_large_scale = torch.abs(torch.log10(param[:, large_scale_stride:]) - torch.log10(param[:, :-large_scale_stride]))
            reg_term_large_scale += torch.sum(weight_diff_large_scale)
    
    # Combine the original loss with the regularization terms
    total_loss = ce_loss + reg_strength * (reg_term_local + 0.2 * reg_term_large_scale)
    return total_loss


def load_progenitors(is50, isDMO, matching_true=None):
    if is50 & isDMO:
        tree1 = np.load('...')
        tree2 = np.load('...')
        ID1 = np.load('...')
        ID2 = np.load('...')  

    elif ~is50 & isDMO:
        tree1 = np.load('...')
        tree2 = np.load('...')
        ID1 = np.load('...')
        ID2 = np.load('...')
        
        
    elif is50 & ~isDMO:
        tree1 = np.load('...')
        tree2 = np.load('...')
        ID1 = np.load('...')
        ID2 = np.load('...')  

    elif ~is50 & ~isDMO:
        tree1 = np.load('...')
        tree2 = np.load('...')
        ID1 = np.load('...')
        ID2 = np.load('...')

    else:
        raise 'Error in load progenitors'
        
    
    h = 0.6774
    tree1[:,:,3] = tree1[:,:,3]/h #[1e10 Msun/h] to [1e10 Msun]
    tree2[:,:,3] = tree2[:,:,3]/h

    if is50:        
        msk1 = tree1[:,-1,3] > 2e-2
        msk2 = tree2[:,-1,3] > 2e-2
    else:
        msk1 = tree1[:,-1,3] > 2e-1
        msk2 = tree2[:,-1,3] > 2e-1

    
    ID1 = ID1[msk1]
    ID2 = ID2[msk2]
    tree1 = tree1[msk1]
    tree2 = tree2[msk2]


    # ----  Mock low resolution-------
    if is50:
        tree1[tree1[:,:,3]<1.5e-2] = np.array([np.nan]*4)
        tree2[tree2[:,:,3]<1.5e-2] = np.array([np.nan]*4)
    else:
        tree2[tree2[:,:,3]<1.5e-1] = np.array([np.nan]*4)
        tree1[tree1[:,:,3]<1.5e-1] = np.array([np.nan]*4)
    # --------------------------------

    #if a matching catalog is provided, remove the subhalos accordingly
    #(default): we don't use the catalog
    if matching_true is not None:
        msk_matching1 = np.in1d(matching_true[ID1],ID2)
        ID1 = ID1[msk_matching1] #exclude no matching pair
        tree1 = tree1[msk_matching1]

    return tree1, tree2, ID1, ID2


