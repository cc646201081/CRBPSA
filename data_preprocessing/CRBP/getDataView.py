import os
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

def get_data(protein):

    if os.path.exists('./Datasets/circRNA-RBP/' + protein + '/pair.npz'):
        print(f"已存在pair文件，正在读取！")
        data = np.load('./Datasets/circRNA-RBP/' + protein + '/pair.npz', allow_pickle=True)
        pair_data = data['x']
        label = data['y']
        pair_data = torch.from_numpy(pair_data)
        label = torch.from_numpy(np.array(label))
    else:
        print("不存在pair文件，正在生成！")
        pair_data = []
        label = []
        with open(r'./Datasets/circRNA-RBP/'+protein+'/positive') as f:
            num = 0
            for data in f:
                if '>' not in data:
                    data = data.strip()
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
                        pair_data = torch.from_numpy(mat).unsqueeze(0)
                    else:
                        matt = torch.from_numpy(mat).unsqueeze(0)
                        pair_data = torch.cat((pair_data,matt),0)
                    label.append(1)
                    # num=num+1
                    # print(num)
        with open(r'./Datasets/circRNA-RBP/'+protein+'/negative') as f:
            for data in f:
                if '>' not in data:
                    data = data.strip()
                    mat = np.zeros([len(data), len(data)])
                    for i in range(len(data)):
                        for j in range(len(data)):
                            coefficient = 0
                            for add in range(30):
                                if i - add >= 0 and j + add < len(data):
                                    score = paired(data[i - add], data[j + add])
                                    if score == 0:
                                        break
                                    else:
                                        coefficient = coefficient + score * Gaussian(add)
                                else:
                                    break
                            if coefficient > 0:
                                for add in range(1, 30):
                                    if i + add < len(data) and j - add >= 0:
                                        score = paired(data[i + add], data[j - add])
                                        if score == 0:
                                            break
                                        else:
                                            coefficient = coefficient + score * Gaussian(add)
                                    else:
                                        break
                            mat[[i], [j]] = coefficient

                    matt = torch.from_numpy(mat).unsqueeze(0)
                    pair_data = torch.cat((pair_data, matt), 0)
                    label.append(0)
                    # num = num + 1
                    # print(num)
        label = torch.from_numpy(np.array(label))
        np.savez('./Datasets/circRNA-RBP/' + protein + '/pair.npz', x=pair_data, y=label)

    train_dataset = dict()
    test_dataset = dict()
    np.random.seed(4)
    indexes = np.random.choice(pair_data.shape[0], pair_data.shape[0], replace=False)

    training_idx, test_idx = indexes[:round(((pair_data.shape[0])/10)*8)], indexes[round(((pair_data.shape[0])/10)*8):] #8:2
    train_dataset['x'] = pair_data[training_idx].float()
    train_dataset['y'] = label[training_idx]

    test_dataset['x'] = pair_data[test_idx].float()
    test_dataset['y'] = label[test_idx]

    return train_dataset, test_dataset






