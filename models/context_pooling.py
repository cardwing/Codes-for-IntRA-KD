import torch
import torch.nn as nn

__all__ = ['ASSP', 'ASSP3', 'PSPP']


class ASSP(nn.Module):
    def __init__(self, in_channels, channels, kernel_size=3, dilation_series=[6, 12, 18, 24]):
        super(ASSP, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.feature_dim = channels
        for dilation in dilation_series:
            padding = dilation * int((kernel_size - 1) / 2)
            self.conv2d_list.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(1, len(self.conv2d_list)):
            out += self.conv2d_list[i](x)
        return out


class ASSP3(nn.Module):
    def __init__(self, in_channels, channels=256, kernel_size=3, dilation_series=[6, 12, 18]):
        super(ASSP3, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.bn2d_list = nn.ModuleList()
        self.feature_dim = channels * (len(dilation_series) + 1) + in_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv2d_list.append(nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False))
        self.bn2d_list.append(nn.BatchNorm2d(channels))

        for dilation in dilation_series:
            padding = dilation * int((kernel_size - 1) / 2)
            self.conv2d_list.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False))
            self.bn2d_list.append(nn.BatchNorm2d(channels))
        self.out_conv2d = nn.Conv2d(self.feature_dim, channels, kernel_size=1, stride=1, bias=False)
        self.out_bn2d = nn.BatchNorm2d(channels)
        self.feature_dim = channels

    def forward(self, x):
        outs = []
        input_size = tuple(x.size()[2:4])

        for i in range(len(self.conv2d_list)):
            outs.append(self.conv2d_list[i](x))
            outs[i] = self.bn2d_list[i](outs[i])
            outs[i] = self.relu(outs[i])
        outs.append(nn.functional.upsample(nn.functional.avg_pool2d(x, input_size), size=input_size))

        out = torch.cat(tuple(outs), dim=1)
        out = self.out_conv2d(out)
        out = self.out_bn2d(out)
        out = self.relu(out)
        return out


class PSPP(nn.Module):
    def __init__(self, in_channels, channels=512, scale_series=[10, 20, 30, 60]):
        super(PSPP, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv2d_list = nn.ModuleList()
        self.bn2d_list = nn.ModuleList()
        self.scale_series = scale_series[:]
        self.feature_dim = channels * len(scale_series) + in_channels

        for i in range(len(scale_series)):
            self.conv2d_list.append(nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False))
            self.bn2d_list.append(nn.BatchNorm2d(channels))

        self.out_conv2d = nn.Conv2d(self.feature_dim, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.out_bn2d = nn.BatchNorm2d(channels)
        self.feature_dim = channels

    def forward(self, x):
        outs = []
        input_size = tuple(x.size()[2:4])
        outs.append(x)

        for i in range(len(self.scale_series)):
            shrink_size = (max(1, int((input_size[0] - 1) / self.scale_series[i] + 1)), max(1, int((input_size[1] - 1) / self.scale_series[i] + 1)))

            pad_h = shrink_size[0] * self.scale_series[i] - input_size[0]
            pad_w = shrink_size[1] * self.scale_series[i] - input_size[1]

            pad_hw = (pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2)

            if sum(pad_hw) > 0:
                outs.append(nn.functional.upsample(torch.nn.functional.pad(x, pad_hw, mode='constant', value=0), size=shrink_size, mode='bilinear'))
            else:
                outs.append(nn.functional.upsample(x, size=shrink_size, mode='bilinear'))

            outs[i + 1] = self.conv2d_list[i](outs[i + 1])
            outs[i + 1] = self.bn2d_list[i](outs[i + 1])
            outs[i + 1] = self.relu(outs[i + 1])
            outs[i + 1] = nn.functional.upsample(outs[i + 1], scale_factor=self.scale_series[i], mode='bilinear')
            outs[i + 1] = outs[i + 1][:, :, pad_h // 2:pad_h // 2 + input_size[0], pad_w // 2:pad_w // 2 + input_size[1]]

        out = torch.cat(tuple(outs), dim=1)
        out = self.out_conv2d(out)
        out = self.out_bn2d(out)
        out = self.relu(out)

        return out
