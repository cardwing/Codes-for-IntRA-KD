import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['FCResNet', 'fc_resnet101']

model_urls = {'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'} # {'resnet101': '/home/xxli/.torch/models/resnet101-sense.pth'}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation, padding=1 * dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.95)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.95)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.95)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=1 * dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.95)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=0.95)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FCResNet(nn.Module):
    def __init__(self, block, layers):
        super(FCResNet, self).__init__()
        self.input_mean = [103.939, 116.779, 123.68]
        self.input_std = [1, 1, 1]
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.95)
        '''self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64, momentum=0.95)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(64, momentum=0.95)
        self.conv1_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(128, momentum=0.95)'''
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64 # 128
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.mid_feature_dim = self.inplanes
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.feature_dim = self.inplanes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, mg=None):
        downsample = None
        if mg is None:
            mg = [1 for _ in range(blocks)]

        if stride != 1 or dilation != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.95), )

        layers = []

        layers.append(block(self.inplanes, planes, stride, dilation * mg[0], downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation * mg[i]))

        return nn.Sequential(*layers)

    def forward(self, x, mid_feature=False):
        '''x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu(x)
        x = self.maxpool(x)'''

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if mid_feature:
            y = self.layer4(x)
            return y, x
        else:
            x = self.layer4(x)
            return x

    def load_state_dict(self, state_dict):
        state_dict.popitem()
        state_dict.popitem()
        super(FCResNet, self).load_state_dict(state_dict, strict=False)


def fc_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FCResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model
