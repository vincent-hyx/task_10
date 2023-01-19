import csv

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

"""
path = 'data/train.csv'
with open(path, 'r', encoding='UTF-8') as f:
    reader = csv.reader(f)
    i = 0
    for line in tqdm(reader):
        if reader.line_num == 1:
            continue
        print(line)
        if i > 2:
            break
        i += 1

with open('data/class.txt', 'r', encoding='UTF-8') as f:
    for line in tqdm(f):
        print(line)

a_list = [([1], [2]), ([2], [3]), ([5], [6]), ([9], [10])]
a_list = np.array(a_list)
print(a_list)
index = [i for i in range(len(a_list))]
np.random.shuffle(index)
random_list = a_list[index]
print(random_list)
a = torch.randn(4, 4)
print(a)
print(torch.max(a, 1))
print(torch.max(a, 1)[1])
a = torch.tensor([[3, 2]])
print(a[0, 1])
a = torch.tensor([[2,3],
                  [1,1],
                  [7,8],
                  [2,2]])
b = torch.reshape(a, [-1,4])
print(b[:, :2])
print(b[:, 2:])
print(b)

a = torch.tensor([[0.1, 0.9],
                  [0.7, 0.3],
                  [0.85, 0.15]])
b = torch.max(a, dim=-1)[0] > 0.8
print(b)
print(torch.sum(b * 0.5, dim=-1))
from scipy.linalg import ldl
a = np.array([[4, -1, 1], [-1, 4.25, 2.75], [1, 2.75, 3.5]])
lu, d, perm = ldl(a, lower=0) # Use the upper part
print(lu)
print(d)
"""
a = torch.tensor([[0.7, 0.3],
            [0.8, 0.2]])
b = torch.tensor([[0,1],[1,0]], dtype=torch.float)
print(F.cross_entropy(a, b, reduction='none'))



