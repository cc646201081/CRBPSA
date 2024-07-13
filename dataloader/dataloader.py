import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from data_preprocessing.CRBP.getDataView import get_data


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()

        self.x = dataset["x"]
        self.y = dataset["y"].long()

    def __getitem__(self, index):

        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def data_generator(protein, configs):

    train_dataset, test_dataset = get_data(protein)



    train_dataset = Load_Dataset(train_dataset)
    test_dataset = Load_Dataset(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=True, # 去掉末尾不够batch_size的样本configs.drop_last
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=True, drop_last=True, # 去掉末尾不够batch_size的样本configs.drop_last
                                              num_workers=0)


    return train_loader, test_loader
