from random import shuffle

import numpy as np
import torch

b = np.array([[1,2,3],[4,5,6],[7,8,9]])

b = torch.FloatTensor(b)
b_n = torch.sqrt(torch.sum(b ** 2, dim=1,keepdim=True))

c = [1,3,4,5]
shuffle(c)
print(c)

