
# Merger Tree-based Galaxy Matching

Official code for 'Merger Tree-based Galaxy Matching: A Comparative Study Across Different Resolutions'.


## 1. Subhalo Matching Between the Simulations

Merger tree and Group catalogs of the TNG simulations are required. 

1. `cd SubhaloMatching`
2. Run `python preprocessing_tree.py` to save main progenitors
3. Run `python parameter_train-valid.py --batch 32 --num_worker 48 --eval_every 2 --tng50` for training
4. Run `python matching.py --tng50` to apply the model to the target galaxy

You can create matching catalogs using the code snippets in `extract_ids.py`. The pre-trained state files are located in the `models` folder.


## 2. Machine Learning Correction

1. `cd MachineLearningCorrection`
2. `python preprocessing_tree_ML.py` 
3. `python save_train_test_split.py` 
4. `python hyper_tuning-100-ablation.py --feat 1 --snap 0 --clf` (for the classification model predicting the stellar mass)
5. `python hyper_tuning-100-ablation.py --feat 1 --snap 0` (for the regression model predicting the stellar mass)

## 3. Datasets

#### TNG subhalo matching catalog between high- and low-resolution runs
See [Jung+ (2023)](https://ui.adsabs.harvard.edu/abs/2023arXiv231202466J/abstract) for more details. [matching_TNG.hdf5](./Dataset/matching_TNG.hdf5) 

| Field | Description                                                                                                                                                      |
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| id1   | SubhaloID in TNG100-1/TNG50-1                                                                                                                                    |
| id2   | SubhaloID in TNG100-2/TNG50-2                                                                                                                                    |
| prob1 | Probability of matching prediction from TNG100-1/TNG50-1 to TNG100-2/TNG50-2                                                                                     |
| prob2 | Probability of matching prediction from TNG100-1/TNG50-1 to TNG100-2/TNG50-2                                                                                     |
| flag  | We recommend excluding the subhalo pair with flag=0. Subhalos with M_DM > 3e9 Msun (or 3e8 Msun for TNG50) and min(prob1, prob2) > 0.7 are marked as one. |


#### Corrected subhalo properties for TNG300-1
See [Jung+ (2023)](https://ui.adsabs.harvard.edu/abs/2023arXiv231202466J/abstract) for more details. 

The corrected properties for the three fields: 'SubhaloMassType0', 'SubhaloMassType1', and 'SubhaloGasMetallicity' are available. All the subhalos with M_DM > 3e9 Msun are included. The units are the same as those used for the corresponding fields in the TNG collaboration.

[TNG300_subhalo_correction_0.hdf5](./Dataset/TNG300_subhalo_correction_0.hdf5), [TNG300_subhalo_correction_1.hdf5](./Dataset/TNG300_subhalo_correction_1.hdf5), [TNG300_subhalo_correction_2.hdf5](./Dataset/TNG300_subhalo_correction_2.hdf5)




## 4. Citation
---
If you find this work useful please cite our paper:

```bibtex
@ARTICLE{2023arXiv231202466J,
       author = {{Jung}, Minyong and {Kim}, Ji-hoon and {Kiat Oh}, Boon and {Hong}, Sungwook E. and {Lee}, Jaehyun and {Kim}, Juhan},
        title = "{Merger Tree-based Halo/Galaxy Matching Between Cosmological Simulations with Different Resolutions: Galaxy-by-galaxy Resolution Study and the Machine Learning-based Correction}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies},
         year = 2023,
        month = dec,
          eid = {arXiv:2312.02466},
        pages = {arXiv:2312.02466},
          doi = {10.48550/arXiv.2312.02466},
archivePrefix = {arXiv},
       eprint = {2312.02466},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv231202466J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
