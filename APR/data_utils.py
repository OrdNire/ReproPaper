import pandas as pd
import config
import numpy as np
from torch.utils.data import Dataset, DataLoader

def load_all_data():
    '''
    need:
    userId -> [itemId]      train/test_user_items       user interaction
    [(userId, itemId)]      train_pair                  interaction pair
    user_num                                            user number
    item_num                                            item number
    '''

    train_data = pd.read_csv(config.TRAIN_RATING, sep='\t', header=None, names=["user", "item"],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    test_data = pd.read_csv(config.TEST_RATING, sep='\t', header=None, names=["user", "item"],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    user_num = max(train_data["user"].max(), test_data["user"].max()) + 1
    item_num = max(train_data["item"].max(), test_data["item"].max()) + 1

    train_user_items = [list() for _ in range(user_num)]
    test_user_items = [list() for _ in range(user_num)]

    for userID, itemID in train_data.values.tolist():
        train_user_items[userID].append(itemID)

    for userID, itemID in test_data.values.tolist():
        test_user_items[userID].append(itemID)

    train_pair = []
    for userID in range(user_num):
        train_pair.extend(zip([userID]*len(train_user_items[userID]), train_user_items[userID]))

    test_pair = []
    for userID in range(user_num):
        test_pair.extend(zip([userID]*len(test_user_items[userID]), test_user_items[userID]))

    return train_user_items, test_user_items, train_pair, test_pair, user_num, item_num

class APRDataset(Dataset):
    def __init__(self, user_num, item_num, user_items, pair, is_training=True):
        super(APRDataset, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.pair = pair
        self.user_items = user_items
        self.is_training = is_training

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        u = self.pair[idx][0]
        i = self.pair[idx][1]
        if self.is_training:
            j = np.random.randint(self.item_num)
            while j in self.user_items[u]:
                j = np.random.randint(self.item_num)

            return u, i, j

        return u, i

if __name__ == '__main__':
    train_user_items, test_user_items, train_pair, test_pair, user_num, item_num = load_all_data()
    dataset = APRDataset(user_num, item_num, train_user_items, train_pair)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)