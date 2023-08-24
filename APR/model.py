import torch.nn as nn
import torch
import torch.nn.functional as F
import data_utils

class AMF(nn.Module):
    def __init__(self, user_num, item_num, emb_size,
                 lambda_param, lambda_adv, epsilon,
                 epoch_adv = 0):
        '''

        :param user_num:        user number
        :param item_num:        item number
        :param emb_size:        embedding size
        :param lambda_param:    proportion of regularisation
        :param lambda_adv:      proportion of adversarial training
        :param epsilon:
        :param epoch_adv:       timing of adversarial training
        '''
        super(AMF, self).__init__()

        # embedding layer
        self.embed_user = nn.Parameter(torch.empty(user_num, emb_size))
        self.embed_item = nn.Parameter(torch.empty(item_num, emb_size))
        nn.init.xavier_uniform_(self.embed_user.data)
        nn.init.xavier_uniform_(self.embed_item.data)

        self.user_num = user_num
        self.item_num = item_num
        self.emb_size = emb_size
        self.lambda_param = lambda_param
        self.lambda_adv = lambda_adv
        self.epsilon = epsilon
        self.epoch_adv = epoch_adv

    def forward(self, user, item_i, item_j, epoch):

        # dim: (batch_size, emb_size)
        user = self.embed_user[user, :]
        item_i = self.embed_item[item_i, :]
        item_j = self.embed_item[item_j, :]

        # keep grad after calculate "delta adv"
        user.retain_grad()
        item_i.retain_grad()
        item_j.retain_grad()
        x_ui = torch.mul(user, item_i).sum(dim=1)
        x_uj = torch.mul(user, item_j).sum(dim=1)

        # calculate BPR loss
        x_uij = torch.clamp(x_ui - x_uj, min=-80.0, max=1e8)
        log_prob = F.logsigmoid(x_uij).sum()
        regularization = self.lambda_param * (user.norm(dim=1).pow(2).sum() + item_i.norm(dim=1).pow(2).sum() + item_j.norm(dim=1).pow(2).sum())
        loss = -log_prob + regularization


        # calculate AMF loss
        loss.backward(retain_graph=True)
        grad_u = user.grad
        grad_i = item_i.grad
        grad_j = item_j.grad

        if grad_u is not None:
            delta_u = self.epsilon * nn.functional.normalize(grad_u, p=2, dim=1)
        else:
            delta_u = torch.rand(user.size())
        if grad_i is not None:
            delta_i = self.epsilon * nn.functional.normalize(grad_i, p=2, dim=1)
        else:
            delta_i = torch.rand(item_i.size())
        if grad_j is not None:
            delta_j = self.epsilon * nn.functional.normalize(grad_j, p=2, dim=1)
        else:
            delta_j = torch.rand(item_j.size())

        x_ui_adv = torch.mul(user + delta_u, item_i + delta_i).sum(dim=1)
        x_uj_adv = torch.mul(user + delta_u, item_j + delta_j).sum(dim=1)
        x_uij_adv = torch.clamp(x_ui_adv - x_uj_adv, min=-80.0, max=1e8)
        log_prob_adv = F.logsigmoid(x_uij_adv).sum()
        adv_loss = loss - self.lambda_adv * log_prob_adv

        if self.epoch_adv == 0:         # train the MF
            return loss
        elif self.epoch_adv is None:    # train the AMF
            return adv_loss
        else:                           # train the MF if epoch in [0, epoch_adv) else AMF
            if epoch < self.epoch_adv:
                return loss
            return adv_loss

if __name__ == '__main__':
    train_user_items, test_user_items, train_pair, test_pair, user_num, item_num = data_utils.load_all_data()
    dataset = data_utils.APRDataset(user_num, item_num, train_user_items, train_pair)
    dataloader = data_utils.DataLoader(dataset, batch_size=64, shuffle=False)
    model = AMF(user_num, item_num, emb_size=64, lambda_param=0, lambda_adv=1, epsilon=0.5, epoch_adv=1000).cuda()
    for u, i, j in dataloader:
        loss = model(u, i, j, 1050)
        print(loss)
        break