import torch.nn as nn
import torch
import torch.nn.functional as F

#inpuut: (B, C, H, W)
#flow:   (B, 2, H, W)
#outputL (B, C, H, W)
def wrap(input, flow):
    device = torch.device(input.device)

    B, C, H, W = input.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(device) # (H, W)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(device)
    # to (B, 1, H, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow

    # 缩放
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input, vgrid)
    return output


class TrajGRU(nn.Module):
    def __init__(self, shape, input_channel, num_features, kernel_size, num_links):
        super(TrajGRU, self).__init__()

        self.shape = shape
        self.input_channel = input_channel
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.num_features = num_features
        self.num_links = num_links
        self.padding = (kernel_size - 1) // 2

        # 对于输入的卷积核
        self.conv_x = nn.Conv2d(in_channels=self.input_channel,
                                out_channels=3 * self.num_features,
                                kernel_size=self.kernel_size,
                                padding=self.padding)

        # generating output (B, 2*L, H, W)
        self.generating = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel + self.num_features,
                      out_channels=32,
                      kernel_size=(5, 5),
                      padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32,
                      out_channels=2*self.num_links,
                      kernel_size=(5, 5),
                      padding=2)
        )

        # 1*1 卷积
        self.conv_h = nn.Conv2d(in_channels=self.num_features * self.num_links,
                                out_channels=3 * self.num_features,
                                kernel_size=(1, 1))

    # input_size (T, B, C, H, W)
    # hidden_state: (B, num_features, H, W)
    # output_inner (T, B, num_features, H, W)
    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        if hidden_state is None:
            device = torch.device(inputs.device)
            h = torch.zeros((inputs.size(1), self.num_features, self.shape[0], self.shape[1])).to(device)
        else:
            h = hidden_state
            device = torch.device(h.device)

        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros((h.size(0), self.input_channel, self.shape[0], self.shape[1])).to(device)
            else:
                x = inputs[index, ...]
            combined = torch.cat([x, h], dim=1)
            uv = self.generating(combined) # (B, 2*L, H, W)
            u, v = torch.split(uv, self.num_links, dim=1)

            # 对于H_t-1的操作
            h_wrap_inner = []
            for l in range(self.num_links):
                u_l = u[:, 1, :, :].unsqueeze(dim=1)
                v_l = v[:, 1, :, :].unsqueeze(dim=1)
                h_wrap = wrap(h, torch.concat([u_l, v_l], dim=1)) # (B, num_features, H, W)
                h_wrap_inner.append(h_wrap)
            h_conv = torch.cat(h_wrap_inner, dim=1) # (B, num_features*num_links, H, W)
            h_out = self.conv_h(h_conv) # (B, num_features*3, H, W)

            hz, hr, hh = torch.split(h_out, self.num_features, dim=1)

            # 处理输入x
            x_out = self.conv_x(x)
            xz, xr, xh = torch.split(x_out, self.num_features, dim=1)

            zt = torch.sigmoid(xz + hz)
            rt = torch.sigmoid(xr + hr)
            ht_t = F.leaky_relu(xh + rt * hh)
            h_n = (1.0 - zt) * ht_t + zt * h

            output_inner.append(h_n)
            h = h_n

        return torch.stack(output_inner), h_n


if __name__ == '__main__':
    seq_len = 10
    batch_size = 4
    channel = 3
    h = 64
    w = 64
    input_tensor = torch.rand((seq_len, batch_size, channel, h, w)).cuda()
    hidden_state = torch.rand((batch_size, 12, h, w)).cuda()
    cell = TrajGRU((64, 64), input_channel=channel, num_features=12, kernel_size=3, num_links=13).cuda()
    out, hidden_state = cell(None,hidden_state,seq_len=seq_len)
    print(out.shape, hidden_state.shape)
