import numpy as np
with open("data/train.txt", 'r') as file_in:
    lst = []
    max_u = 0
    max_i = 0
    for line in file_in:
        uir = [int(i) for i in line.strip().split('\t')]
        lst.append(uir)
        if uir[0] > max_u:
            max_u = uir[0]
        if uir[1] > max_i:
            max_i = uir[1]
    R = np.zeros((max_u, max_i))
    for uir in lst:
        R[uir[0] - 1, uir[1] - 1] = uir[2]
np.save("data/train", R)
