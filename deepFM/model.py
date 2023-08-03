import torch.nn as nn
import torch
import config
from datasets import DeepFMDataset
from torch.utils.data import DataLoader
import pandas as pd

class DeepFM(nn.Module):
    def __init__(self, features_size ,continuous_features, categorial_features, emb_size=10, num_neurons=400, dropout=0.5, num_hidden=3):
        super(DeepFM, self).__init__()

        self.continuous_features = continuous_features
        self.categorial_features = categorial_features

        # FM component
        self.embds = [nn.Linear(features_size[idx], emb_size) for idx in continuous_features]
        self.embds += [nn.Embedding(features_size[idx], emb_size) for idx in categorial_features]
        self.embds = nn.ModuleList(self.embds)
        self.linear_part = nn.Linear(len(features_size), 1)
        # deep component
        input_size = emb_size * len(features_size)
        modules = []
        for i in range(num_hidden):
            modules.append(nn.Linear(input_size, num_neurons))
            modules.append(nn.BatchNorm1d(num_neurons))
            modules.append(nn.Dropout(p=dropout))
            modules.append(nn.ReLU())
            input_size = num_neurons
        modules.append(nn.Linear(num_neurons, 1))
        self.deepComponent = nn.Sequential(*modules)

    def forward(self, features):

        deep_input = []
        ### FM component ###
        results = []
        # FM order-2 part
        for i, emb in enumerate(self.embds):
            batch_idx = torch.Tensor(features[:, i])

            if i in self.continuous_features:
                batch_idx = batch_idx.unsqueeze(1)
            else:
                batch_idx = batch_idx.int()

            batch_v = emb(batch_idx)
            deep_input.append(batch_v)
            results.append(batch_v)

        fm1 = sum(results)
        fm1 = fm1 * fm1
        fm1 = torch.sum(fm1, dim=1)

        fm2 = [item * item for item in results]
        fm2 = sum(fm2)
        fm2 = torch.sum(fm2, dim=1)
        quadratic_term = ((fm1 - fm2) * 0.5).unsqueeze(1)
        # FM order-1 part
        linear_term = self.linear_part(features)

        fm_component = quadratic_term + linear_term

        ### deep component ###
        deep_input = torch.concat(deep_input, dim=1)
        deep_component = self.deepComponent(deep_input)

        return (fm_component + deep_component).view(-1)

if __name__ == '__main__':
    deepFMdataset = DeepFMDataset(config.HANDLE_TRAINING_PATH)
    loader = DataLoader(deepFMdataset, batch_size=3, shuffle=False)
    features_size = pd.read_csv(config.HANDLE_FEATURES_SIZE_PATH, sep=',', header=None)
    features_size = list(features_size.values.squeeze(0))
    continuous_features = range(0, 13)
    categorial_features = range(13, 39)

    model = DeepFM(features_size, continuous_features, categorial_features)
    for data, label in loader:
        predict = model(data)
        print(predict)
        break