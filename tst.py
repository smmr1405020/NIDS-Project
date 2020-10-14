from random import shuffle

import numpy as np
import torch

b = np.array([[1,2,3],[4,5,6]]) # 2x3
c = np.array([[7,8,9],[10,11,12],[13,14,15],[16,17,18]]) # 4x3

c_exp = np.expand_dims(c,axis=1)
print(c_exp.shape)
c_exp_repeated = np.repeat(c_exp,b.shape[0],axis=1)
print(c_exp_repeated)

b_exp = np.expand_dims(b,axis=0)
b_exp_repeated = np.repeat(b_exp,c.shape[0],axis=0)

print(np.sqrt(np.sum((c_exp_repeated - b_exp_repeated) ** 2, axis=2)).shape)

