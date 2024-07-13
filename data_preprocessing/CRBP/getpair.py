import os.path

import numpy as np
import collections
import math
import torch
def Gaussian(x):
    return math.exp(-0.5*(x*x))
def paired(x,y):
    if x == 'A' and y == 'U':
        return 2
    elif x == 'G' and y == 'C':
        return 3
    elif x == "G"and y == 'U':
        return 0.8
    elif x == 'U' and y == 'A':
        return 2
    elif x == 'C' and y == 'G':
        return 3
    elif x == "U"and y == 'G':
        return 0.8
    else:
        return 0
def creatmat(filename):
    if os.path.exists(filename+'.npz'):
        print(f"已存在pair文件，正在读取！")
        data = np.load(filename+'.npz', allow_pickle=True)
        x = data['x']
        y = data['y']
        s = data['s']
        return x, y, s
    print("不存在pair文件，正在生成！")
    pair_data = []
    label = []
    seq = []
    with open(filename) as f:
        num = 0
        for data in f:
            if '>' not in data:
                data = data.strip()
                seq.append(data)
                mat = np.zeros([len(data),len(data)])
                for i in range(len(data)):
                    for j in range(len(data)):
                        coefficient = 0
                        for add in range(30):
                            if i - add >= 0 and j + add <len(data):
                                score = paired(data[i - add].replace('T', 'U'),data[j + add].replace('T', 'U'))
                                if score == 0:
                                    break
                                else:
                                    coefficient = coefficient + score * Gaussian(add)
                            else:
                                break
                        if coefficient > 0:
                            for add in range(1,30):
                                if i + add < len(data) and j - add >= 0:
                                    score = paired(data[i + add],data[j - add])
                                    if score == 0:
                                        break
                                    else:
                                        coefficient = coefficient + score * Gaussian(add)
                                else:
                                    break
                        mat[[i],[j]] = coefficient
                if len(pair_data)==0:
                    pair_data = np.expand_dims(mat, axis=0)
                else:
                    matt = np.expand_dims(mat, axis=0)
                    pair_data = np.cat((pair_data,matt),0)
                label.append(1)

    label = torch.from_numpy(np.array(label))
    return pair_data, label, seq

