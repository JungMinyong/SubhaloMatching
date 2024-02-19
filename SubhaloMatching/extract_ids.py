import numpy as np

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
