import torch
from torch import nn
import torch.nn.init as init

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResfftBlock(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'ortho'
        super(ResfftBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=True),
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=True),
            BasicConv(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=False)
        )
        self.dim = n_feat
        self.norm = norm
        self._initialize_weights()
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print(f'init weight {m}')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# model = ResfftBlock(3)
# print(model)

# input = torch.randn(1,3,572,572)
# out = model(input)
# print(out.shape)