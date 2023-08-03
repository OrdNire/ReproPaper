import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, layer_num, dropout, model, GMF_model=None, MLP_model=None):
        '''
        :param user_num:        number of users
        :param item_num:        number of items
        :param factor_num:      number of predictive factors
        :param layer_num:       number of mlp layers
        :param dropout:         dropout ratio
        :param model:           "GMF", "MLP", "NeuMF-end", "NeuMF-pre"
        :param GMF_model:       pre-trained GMF weights;
        :param MLP_model:       pre-trained MLP weights;
        '''
        super(NCF, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.factor_num = factor_num
        self.layer_num = layer_num
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        assert self.model in ["GMF", "MLP", "NeuMF-end", "NeuMF-pre"]

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(user_num, (2**(layer_num - 1)) * factor_num)
        self.embed_item_MLP = nn.Embedding(item_num, (2**(layer_num - 1)) * factor_num)

        MLP_modules = []
        for i in range(layer_num):
            input_size = (2**(layer_num - i)) * factor_num
            MLP_modules.append(nn.Dropout(p=dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layer = nn.Sequential(*MLP_modules)

        if self.model in ["GMF", "MLP"]:
            predict_size = factor_num
        else:
            predict_size = 2 * factor_num

        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        if not self.model == "NeuMF-pre":
            nn.init.normal_(self.embed_user_GMF.weight, std=0.1)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.1)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.1)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.1)

            for m in self.MLP_layer:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

            nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity="sigmoid")

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)

            for (m1, m2) in zip(self.MLP_layer, self.MLP_model.MLP_layer):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight,
                self.MLP_model.predict_layer.weight
            ], dim=1)

            predict_bias = self.GMF_model.predict_layer.bias + self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * predict_bias)


    def forward(self, user, item):
        if not self.model == "GMF":
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat([embed_user_MLP, embed_item_MLP], dim=-1)
            output_MLP = self.MLP_layer(interaction)
        if not self.model == "MLP":
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF

        if self.model == "GMF":
            concat = output_GMF
        elif self.model == "MLP":
            concat = output_MLP
        else:
            concat = torch.concat([output_GMF, output_MLP], dim=-1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

if __name__ == '__main__':
    from data_utils import *
    train_data, test_data, user_num, item_num, train_matrix = load_all_data()
    ncfdataset = NCFDataset(train_data, item_num, train_matrix, num_ng=4, is_training=train_data)
    ncfdataset.ng_sample()
    dataloader = DataLoader(ncfdataset, batch_size=128, shuffle=True)
    model = NCF(user_num, item_num, factor_num=8, layer_num=3, dropout=0.2, model="MLP")
    for user, item, label in dataloader:
        prediction = model(user, item)
        print(prediction)