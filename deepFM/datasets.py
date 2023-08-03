from torch.utils.data import Dataset, DataLoader
import pandas as pd
import config
import numpy as np

class DeepFMDataset(Dataset):
    def __init__(self, root, is_training=True):
        super(DeepFMDataset, self).__init__()

        self.root = root
        self.is_training = is_training

        # load data
        self.data = pd.read_csv(root, sep=',', header=None).values.astype(np.float32)

    def __getitem__(self, idx):
        if self.is_training:
            features, target = self.data[idx, :-1], self.data[idx, -1]
            return features, target
        else:
            features = self.data[idx, :]
            return features

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    deepFMdataset = DeepFMDataset(config.HANDLE_TRAINING_PATH)
    loader = DataLoader(deepFMdataset, batch_size=64, shuffle=False)
    for data, label in loader:
        print(data, label)
        break