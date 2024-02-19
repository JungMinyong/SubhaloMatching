#Extract main progentors of all subhalos in TNG simulations.
#Save it to numpy array

import sys
import h5py
import numpy as np
import pandas as pd
import illustris_python as il

#BasePath of the TNG data
path = ".../TNG/TNG50-2/output"

mass = il.groupcat.loadSubhalos(path, 99, "SubhaloMassType")[:, 1]
pos = il.groupcat.loadSubhalos(path, 99, "SubhaloPos")
ID_total = np.arange(len(mass))

#exclude subhalos with no DM particles
msk_mass = (mass > 0)

ID = ID_total[msk_mass]
dtype = np.float32
data1 = np.zeros((len(ID), 100, 4), dtype=dtype) * np.nan

for i, id in enumerate(ID):
    tree_main = il.sublink.loadTree(
        path,
        99,
        id,
        fields=["SnapNum", "SubhaloPos", "SubhaloMassType"],
        onlyMPB=True,
    )
    data1[i, :, :3][tree_main["SnapNum"]] = tree_main["SubhaloPos"]
    data1[i, :, 3][tree_main["SnapNum"]] = tree_main["SubhaloMassType"][:, 1]
    if i%10000 == 0:
        print(i, len(ID))

        
#Save the numpy array
np.save('TNG50-2-mainbranch-subset',data1)
np.save('TNG50-2-mainbranch-subset-ID',ID)
