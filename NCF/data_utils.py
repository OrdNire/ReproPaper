import pandas as pd
import config
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader

def load_all_data():
    '''
    load all data
    :return:
    '''
    train_data = pd.read_csv(config.TRAIN_RATING,
                             sep="\t", header=None, names=["user", "item"],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    user_num_max, item_num_max = train_data["user"].max(), train_data["item"].max()

    train_data = train_data.values.tolist()

    test_data = []
    with open(config.TEST_NEGATIVE, 'r') as f:
        line = f.readline()
        while line is not None and line != '':
            arr = line.split("\t")
            u = eval(arr[0])[0]
            user_num_max = max(u, user_num_max)
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
                item_num_max = max(item_num_max, int(i))
            line = f.readline()

    # user number and item number
    user_num = user_num_max + 1
    item_num = item_num_max + 1

    # sparse matrix
    train_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)

    for row in train_data:
        train_matrix[row[0], row[1]] = 1.0

    return train_data, test_data, user_num, item_num, train_matrix

class NCFDataset(Dataset):
    def __init__(self, data, item_num, train_matrix = None, num_ng=0, is_training=None):
        '''
        NCFDataset
        :param data:            training data/test data
        :param item_num:        item number
        :param train_matrix:    train matrix
        :param num_ng:          negative sample ratio (1 : num_ng)
        :param is_training:     train or test
        '''
        super(NCFDataset, self).__init__()

        self.data_ps = data
        self.item_num = item_num
        self.num_ng = num_ng
        self.train_matrix = train_matrix
        self.is_training = is_training
        self.labels = [0 for _ in range(len(self.data_ps))]

    # negative sample (1:num_ng)
    def ng_sample(self):
        assert self.is_training     # no need to sampling when testing

        self.data_ng = []
        for row in self.data_ps:
            u = row[0]
            for i in range(self.num_ng):
                # sample negative data
                j = np.random.randint(self.item_num)
                while (u, j) in self.train_matrix:
                    j = np.random.randint(self.item_num)
                self.data_ng.append([u, j])

        self.label_ps = [1 for _ in range(len(self.data_ps))]
        self.label_ng = [0 for _ in range(len(self.data_ng))]

        self.data_fill = self.data_ps + self.data_ng
        self.label_fill = self.label_ps + self.label_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, index):
        data = self.data_fill if self.is_training else self.data_ps
        labels = self.label_fill if self.is_training else self.labels

        user = data[index][0]
        item = data[index][1]
        label = labels[index]

        return user, item, label

if __name__ == '__main__':
    train_data, test_data, user_num, item_num, train_matrix = load_all_data()
    ncfdataset = NCFDataset(test_data, item_num, train_matrix, num_ng=0, is_training=False)
    dataloader = DataLoader(ncfdataset, batch_size=100, shuffle=False)
    for user, item, label in dataloader:
        print(item.max())
        if item.max() == 3706:
            break