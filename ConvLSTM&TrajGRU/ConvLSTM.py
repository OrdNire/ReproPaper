import torch
import torch.nn as nn

# class ConvLSTMCell(nn.Module):
#     def __init__(self, input_size, hidden_size, kernel_size, bias):
#         '''
#         :param input_size:  输入通道数
#         :param hidden_size: 隐藏层通道数
#         :param kernel_size: 卷积核大小
#         :param bias:        偏置 True/False
#         '''
#         super(ConvLSTMCell, self).__init__()
#
#         self.input_size= input_size
#         self.hidden_size = hidden_size
#         self.kernel_size = kernel_size
#         self.bias = bias
#
#         self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2
#
#         # [W1, W2] * [X, H].T -> 4 * [hidden_size] 4个等式一起算
#         self.conv = nn.Conv2d(in_channels=input_size + hidden_size,
#                               out_channels=4*hidden_size,
#                               kernel_size=kernel_size,
#                               padding=self.padding,
#                               bias=self.bias)
#
#     #cur_state = (h, c)
#     #input_tensort (B, C, H, W)
#     def forward(self, input_tensor, cur_state):
#         h_cur, c_cur = cur_state
#
#         combined = torch.cat([input_tensor, h_cur], dim=1)
#         # 在 Channel 维度concat
#         combined_conv = self.conv(combined)
#         xh_i, xh_f, xh_g, xh_o = torch.split(combined_conv, split_size_or_sections=self.hidden_size, dim=1)
#         i = torch.sigmoid(xh_i)
#         f = torch.sigmoid(xh_f)
#         o = torch.sigmoid(xh_o)
#         c = f * c_cur + i * torch.tanh(xh_g)
#         h = o * torch.tanh(c)
#
#         return h, c
#
#     # return (h0, c0)
#     def init_hidden(self, batch_size, image_size):
#         h, w = image_size
#         return (torch.zeros(batch_size, self.hidden_size, h, w, device=self.conv.weight.device),
#                 torch.zeros(batch_size, self.hidden_size, h, w, device=self.conv.weight.device))

# class ConvLSTM&TrajGRU(nn.Module):
#     def __init__(self, input_size, hidden_size, kernel_size, num_layers=1, bias=True, batch_first=False):
#         '''
#         输出与输出与LSTM文档一致
#         :param input_size:   输入图像通道数
#         :param hidden_size:  隐层通道数
#         :param kernel_size:  must be (tuple, tuple lists)
#         :param num_layers:
#         :param bias:
#         :param batch_first: True (B, T, C, H, W) False->(T, B, C, H, W)
#         Input: input, (h0, c0)   input_size=(T, B, C, H, W) or (B, T, C, H, W)
#         Output: output, (hn, cn)
#         output: last_layers per_time output B_first->(B, T, hidden_size, H, W) or (T, B, hidden_size, H, W)
#         hn:     per layers last h  hn(layers_num, B, hidden_size, H, W)
#         cn:     same as hn
#         '''
#         super(ConvLSTM&TrajGRU, self).__init__()
#         self._check_kernel_size(kernel_size)
#
#         self.kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
#         self.hidden_size = self._extend_for_multilayer(hidden_size, num_layers)
#
#         self.input_size = input_size
#         self.num_layers = num_layers
#         self.bias = bias
#         self.batch_first = batch_first
#
#         cell_list = []
#         for i in range(0, num_layers):
#             cur_input_size = self.input_size if i == 0 else self.hidden_size[i - 1]
#             cell_list.append(ConvLSTMCell(cur_input_size, self.hidden_size[i], self.kernel_size[i], self.bias))
#
#         self.cell_list = nn.ModuleList(cell_list)
#
#     def forward(self, input_tensor, hidden_state=None):
#         '''
#
#         :param input_tensor: B_first->(B, T, C, H, W) or (T, B, C, H, W)
#         :param hidden_state: (h_0, c_0))
#         :return:
#         '''
#
#         #(b, t, c, h, w) -> (t, b, c, h, w)
#         if self.batch_first:
#             input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
#         _, b, _, h, w = input_tensor.size()
#
#         if hidden_state is not None:
#             raise NotImplementedError()
#         else:
#             hidden_state = self._init_hidden_state(b, image_size=(h, w))
#
#         output = []
#         h_n = []
#         c_n = []
#
#         seq_len = input_tensor.size(0)
#         cur_layer_input = input_tensor
#
#         for layer_idx in range(self.num_layers):
#             h, c = hidden_state[layer_idx]
#             output_inner = []
#             for t in range(seq_len):
#                 h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[t, :, :, :, :], cur_state=(h, c))
#                 output_inner.append(h)
#
#             layer_output = torch.stack(output_inner, dim=0)
#             cur_layer_input = layer_output
#
#             output.append(layer_output)
#             h_n.append(h)
#             c_n.append(c)
#
#         # output (T, B, C, H, W)
#         output = output[-1]
#         if self.batch_first:
#             output = output.permute(1, 0, 2, 3, 4)
#         # h_n (B, C, H, W)
#         device = next(self.parameters()).device
#         h_n = torch.stack(h_n, dim=0)
#         c_n = torch.stack(c_n, dim=0)
#         return output, h_n, c_n
#
#     def _init_hidden_state(self, batch_size, image_size):
#         return [cell.init_hidden(batch_size, image_size) for cell in self.cell_list]
#
#     @staticmethod
#     def _check_kernel_size(kernel_size):
#         if not (isinstance(kernel_size, tuple) or
#                 (isinstance(kernel_size, list) and all([isinstance(kernel, tuple) for kernel in kernel_size]))):
#             raise ValueError('`kernel_size` must be int, tuple or list of tuples')
#
#     @staticmethod
#     def _extend_for_multilayer(param, num_layers):
#         if not isinstance(param, list):
#             param = [param] * num_layers
#         return param


class ConvLSTMCell(nn.Module):
    def __init__(self, shape, input_channel, kernel_size, num_features):
        super(ConvLSTMCell, self).__init__()

        self.shape = shape
        self.input_channel = input_channel
        self.kernel_size = kernel_size
        self.num_features = num_features
        # keep same
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel + self.num_features,
                      out_channels=4 * self.num_features,
                      kernel_size=self.kernel_size,
                      stride=1,
                      padding=self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features)
        )

    # input_size (T, B, C, H, W)
    # hidden_state -> h, c -> h_size (B, num_features, H, W)
    def forward(self, input=None, hidden_state=None, seq_len=10):
        if hidden_state is None:
            h, c = self._init_hidden_state(input.size(1))
        else:
            h, c = hidden_state

        output_inner = []
        for seq_index in range(seq_len):
            if input is None:
                x = torch.zeros((h.size(0), self.input_channel, self.shape[0], self.shape[1])).cuda()
            else:
                x = input[seq_index, ...]
            combined = torch.cat([x, h], dim=1)
            gates = self.conv(combined) # shape (B, 4 * num_features, H, W)
            i, f, g, o = torch.split(gates, self.num_features, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)

            c_n = (f * c) + (i * g)
            h_n = o * torch.tanh(c_n)
            output_inner.append(h_n)
            h = h_n
            c = c_n

        # output_inner (T, B, num_features, H, W)
        return torch.stack(output_inner), (h_n, c_n)

    def _init_hidden_state(self, batch_size):
        h = torch.zeros((batch_size, self.num_features, self.shape[0], self.shape[1])).cuda()
        c = torch.zeros((batch_size, self.num_features, self.shape[0], self.shape[1])).cuda()
        return h, c


if __name__ == '__main__':
    seq_len = 10
    batch_size = 64
    channel = 3
    h = 64
    w = 64
    input_tensor = torch.rand((seq_len, batch_size, channel, h, w)).cuda()
    cell = ConvLSTMCell((h, w), input_channel=channel, kernel_size=3, num_features=12).cuda()
    out, hidden_state = cell(input_tensor, seq_len=seq_len)
    print(out.shape, hidden_state[0].shape, hidden_state[1].shape)