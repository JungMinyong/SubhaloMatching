import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.neighbors import KDTree
from utils import making_buffer


dtype = torch.float32
np.seterr(invalid='ignore')
dist_plus = 5000 #5 Mpc/h

class CustomDataset(Dataset): 
    def __init__(self, tree1, tree2, ID1, ID2, factor_dmo, L_box, matching_true):
        xyzdata = tree2[:,-1,:3]
        xyzBuffer, indexBuffer = making_buffer(xyzdata, dist_plus, L_box)        
        self.kdtree2 = KDTree(xyzBuffer, leaf_size=400)
        self.indexBuffer = indexBuffer
        self.tree1 = tree1
        self.tree2 = tree2
        self.ID1 = ID1
        self.ID2 = ID2
        self.factor_dmo = factor_dmo
        self.L_box = L_box
        self.matching_true = matching_true
        
    def __len__(self): 
        return len(self.ID1)

    def __getitem__(self, idx):
        L_box = self.L_box
        matching_true = self.matching_true

        id1 = self.ID1[idx]
        index_temp = self.kdtree2.query_radius([self.tree1[:,-1,:3][idx]], dist_plus)[0] #3800
        index = self.indexBuffer[index_temp]

        dist_temp = np.abs(self.tree2[index][:,:,:3] - self.tree1[idx][:,:3])
        msk_temp = dist_temp > L_box/2
        dist_temp[msk_temp] = L_box - dist_temp[msk_temp]
        dists = np.linalg.norm(dist_temp, axis=2)

        masses = np.abs(np.log10(self.tree2[index][:,:,3] * self.factor_dmo / self.tree1[idx][:,3])) #0.8427
        masses_z0 = torch.tensor(self.tree1[idx][-1,3])

        dists = dists / 100 #act as normalization

        dists = torch.tensor(dists, dtype=dtype)
        masses = torch.tensor(masses, dtype=dtype)

        id2 = matching_true[id1]
        if id2 in self.ID2[index]:
            y_true = torch.tensor(np.where(id2==self.ID2[index])[0][0])
            y_true = nn.functional.one_hot(y_true, num_classes=len(index)).to(torch.float)
            #assert self.ID2[index][y_true] == id2, 'mismatch in true matching'

        else: #id2 == -1:
            y_true = torch.ones(len(index), dtype=dtype)/len(index)

        return dists, masses, y_true, masses_z0, id1
    

def custom_collate(batch): #(2)
    dists_lst, masses_lst, y_lst, masses_z0_lst, id1_lst = [], [], [], [], []
    
    for (_dists, _masses, _y, _masses_z0, _id1) in batch:
        dists_lst.append(_dists)
        masses_lst.append(_masses)
        y_lst.append(_y)
        masses_z0_lst.append(_masses_z0)
        id1_lst.append(_id1)
        
    dists = pad_sequence(dists_lst, batch_first=True,padding_value=5000.)
    masses = pad_sequence(masses_lst, batch_first=True,padding_value=100.)
    masses_z0 = torch.tensor(masses_z0_lst)
    y = pad_sequence(y_lst, batch_first=True,padding_value=0.)#torch.tensor(y_lst)
    id1 = torch.tensor(id1_lst)

    return dists, masses, y, masses_z0, id1

class Prediction(torch.nn.Module):
    def __init__(self):
        super(Prediction, self).__init__()

        self.linear = nn.Linear(25, 1)
        self.w_mass = torch.nn.Parameter(torch.tensor(1.0,dtype=dtype))
        self.gamma = torch.nn.Parameter(torch.tensor(1.0,dtype=dtype))
        self.dropout = nn.Dropout(p=0.)
        
        self.p1 = torch.nn.Parameter(torch.tensor(5.0,dtype=dtype))

        self.linear.weight.data.fill_(0.01)
        
    def forward(self, dists, masses, masses_z0):
        #dists: (B, N, 100) shape
        #output: (B, N) shape (w/ squeeze(2))
        
        dists = torch.nan_to_num(dists, nan=0.)
        masses = torch.nan_to_num(masses, nan=0.)
        score = dists + self.w_mass * masses **self.gamma
        score[score==0] = self.p1 
        
        B, N, _ = score.shape
        score = score.reshape(B, N, 25, 4)  # reshape the tensor to size (B, N, 25, 4)
        output = score.sum(dim=3)
        
        final_score = - self.linear(self.dropout(output))
        
        return final_score.squeeze(2)
    
    
def custom_loss(y_pred, y_true, model, reg_strength=1e-4, large_scale_stride=5, device='cuda'):
    # Calculate the original loss (e.g., mean squared error)
    criterion = nn.CrossEntropyLoss().to(device)
    
    ce_loss = criterion(y_pred, y_true)
    
    # Calculate the regularization term for local smoothness
    reg_term_local = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_diff_local = torch.abs(torch.log10(param[:, 1:]) - torch.log10(param[:, :-1]))
            reg_term_local += torch.sum(weight_diff_local)

    # Calculate the regularization term for large-scale smoothness
    reg_term_large_scale = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_diff_large_scale = torch.abs(torch.log10(param[:, large_scale_stride:]) - torch.log10(param[:, :-large_scale_stride]))
            reg_term_large_scale += torch.sum(weight_diff_large_scale)
    
    # Combine the original loss with the regularization terms
    total_loss = ce_loss + reg_strength * reg_term_local #(reg_term_local + 0.01 * reg_term_large_scale)
    return total_loss




class CustomDatasetMatching(Dataset): 
    def __init__(self, tree1, tree2, ID1, ID2, factor_dmo, L_box):
        xyzdata = tree2[:,-1,:3]
        xyzBuffer, indexBuffer = making_buffer(xyzdata, dist_plus, L_box)        
        self.kdtree2 = KDTree(xyzBuffer, leaf_size=400)
        self.indexBuffer = indexBuffer
        self.tree1 = tree1
        self.tree2 = tree2
        self.ID1 = ID1
        self.ID2 = ID2
        self.factor_dmo = factor_dmo
        self.L_box = L_box
        
    def __len__(self): 
        return len(self.ID1)

    def __getitem__(self, idx):
        L_box = self.L_box

        id1 = self.ID1[idx]
        index_temp = self.kdtree2.query_radius([self.tree1[:,-1,:3][idx]], dist_plus)[0] #3800
        index = self.indexBuffer[index_temp]

        dist_temp = np.abs(self.tree2[index][:,:,:3] - self.tree1[idx][:,:3])
        msk_temp = dist_temp > L_box/2
        dist_temp[msk_temp] = L_box - dist_temp[msk_temp]
        dists = np.linalg.norm(dist_temp, axis=2)

        masses = np.abs(np.log10(self.tree2[index][:,:,3] * self.factor_dmo / self.tree1[idx][:,3])) #0.8427
        masses_z0 = torch.tensor(self.tree1[idx][-1,3])

        dists = dists / 100 #act as normalization

        dists = torch.tensor(dists, dtype=dtype)
        masses = torch.tensor(masses, dtype=dtype)

        idx2_to_id2 = torch.tensor(self.ID2[index])

        return dists, masses, idx2_to_id2, masses_z0, id1
    

def custom_collate_matching(batch): #(2)
    dists_lst, masses_lst, id2_lst, masses_z0_lst, id1_lst = [], [], [], [], []
    
    for (_dists, _masses, _id2, _masses_z0, _id1) in batch:
        dists_lst.append(_dists)
        masses_lst.append(_masses)
        id2_lst.append(_id2)
        masses_z0_lst.append(_masses_z0)
        id1_lst.append(_id1)
        
    dists = pad_sequence(dists_lst, batch_first=True,padding_value=5000.)
    masses = pad_sequence(masses_lst, batch_first=True,padding_value=100.)
    masses_z0 = torch.tensor(masses_z0_lst)
    id2_arr = pad_sequence(id2_lst, batch_first=True,padding_value=-1.)#torch.tensor(y_lst)
    id1 = torch.tensor(id1_lst)

    return dists, masses, id2_arr, masses_z0, id1