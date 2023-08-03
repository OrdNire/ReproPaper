import torch
import torch.nn as nn
from utils import make_layers

class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        '''
        :param subnets: come from model_params[0]
        :param rnn:     come from model_params[1]
        '''
        super(Encoder, self).__init__()
        assert len(subnets) == len(rnns)
        self.block_len = len(subnets)

        for i, (param, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, "stage" + str(i), make_layers(param))
            setattr(self, "rnn" + str(i), rnn)

    # input size (T, B, C, H, W)
    def forward_by_stage(self, input, subnet, rnn):
        seq_len, batch_size, input_channel, height, weight = input.size()
        input = torch.reshape(input, (-1, input_channel, height, weight))
        input = subnet(input)
        input = torch.reshape(input, (seq_len, batch_size, input.size(1), input.size(2), input.size(3)))
        outputs_stage, state_stage = rnn(input, None, seq_len=seq_len)
        return outputs_stage, state_stage

    # input size (B, T, C, H, W)
    def forward(self, inputs):
        inputs = inputs.permute(1, 0, 2, 3, 4) # to (T, B, C, H, W)
        hidden_states = []
        for i in range(1, self.block_len + 1):
            inputs, state_stage = self.forward_by_stage(
                inputs,
                getattr(self, "stage" + str(i)),
                getattr(self, "rnn" + str(i))
            )
            hidden_states.append(state_stage)
        return tuple(hidden_states)

class Forecaster(nn.Module):
    def __init__(self, subnets, rnns):
        super(Forecaster, self).__init__()
        assert len(subnets) == len(rnns)
        self.block_len = len(subnets)

        for i, (param, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, "rnn" + str(self.block_len - i), rnn)
            setattr(self, "stage" + str(self.block_len - i), make_layers(param))

    # input size (T, B, C, H, W) or None
    # output size (T, B, C, H, W)
    # rnn -> subnet
    def forward_by_stage(self, input, state, subnet, rnn):
        input, _ = rnn(input, state, seq_len=10)
        seq_len, batch_size, input_channel, height, weight = input.size()
        input = torch.reshape(input, (-1, input_channel, height, weight))
        input = subnet(input)
        input = torch.reshape(input, (seq_len, batch_size, input.size(1), input.size(2), input.size(3)))
        return input

    def forward(self, hidden_states):
        input = self.forward_by_stage(None, hidden_states[-1],
                                      getattr(self, "stage" + str(self.block_len)),
                                      getattr(self, "rnn" + str(self.block_len)))
        for i in list(range(1, self.block_len))[::-1]:
            input = self.forward_by_stage(input, hidden_states[i - 1],
                                          getattr(self, "stage" + str(i)),
                                          getattr(self, "rnn" + str(i)))
        # input (T, B, C, H, W)
        input = input.permute(1, 0, 2, 3, 4) # to (B, T, C, H, W)
        return input

class EF(nn.Module):
    def __init__(self, convlstm_encoder_params, convlstm_forecaster_params):
        super(EF, self).__init__()
        self.encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1])
        self.forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1])

    # inputs (B, T, C, H, W)
    def forward(self, inputs):
        hidden_states = self.encoder(inputs)
        return self.forecaster(hidden_states)

def test_encoder():
    from MovingMNIST import MovingMNIST
    from torch.utils.data import DataLoader
    from model_params import convlstm_encoder_params, trajgru_encoder_params
    train_set = MovingMNIST("datasets/mnist", Norm=True)
    train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True)
    encoder = Encoder(trajgru_encoder_params[0], trajgru_encoder_params[1]).cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        hidden_states = encoder(inputs) # (B, num_features, H, W)
        print(len(hidden_states), hidden_states[0].shape, hidden_states[1].shape, hidden_states[2].shape)
        break

def test_forecaster():
    from MovingMNIST import MovingMNIST
    from torch.utils.data import DataLoader
    from model_params import convlstm_encoder_params, convlstm_forecaster_params, trajgru_forecaster_params, trajgru_encoder_params
    train_set = MovingMNIST("datasets/mnist", Norm=True)
    train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True)
    encoder = Encoder(trajgru_encoder_params[0], trajgru_encoder_params[1]).cuda()
    forecaster = Forecaster(trajgru_forecaster_params[0], trajgru_forecaster_params[1]).cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        hidden_states = encoder(inputs)  # (B, num_features, H, W)
        out = forecaster(hidden_states)  # out (B, T, 1, C, H)
        print(out.shape)
        break

def test_EF():
    from MovingMNIST import MovingMNIST
    from torch.utils.data import DataLoader
    from model_params import convlstm_encoder_params, convlstm_forecaster_params, trajgru_forecaster_params, trajgru_encoder_params
    train_set = MovingMNIST("datasets/mnist", Norm=True)
    train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True)
    ef = EF(trajgru_encoder_params, trajgru_forecaster_params).cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        out = ef(inputs)
        print(out.shape)
        break

if __name__ == '__main__':
    # test_encoder()
    # test_forecaster()
    test_EF()