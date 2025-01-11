import math
import torch
import torch.nn as nn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnet18_d', 'resnet34_d', 'resnet50_d', 'resnet101_d', 'resnet152_d',
           'resnet50_16s', 'resnet50_w2x', 'resnext101_32x8d', 'resnext152_32x8d']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm3d(out_planes),
        nn.ReLU()
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=-1):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv3d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(width)
        self.conv2 = nn.Conv3d(width, width, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm3d(width)
        self.conv3 = nn.Conv3d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=1, width=1,
                 groups=1, width_per_group=64,
                 mid_dim=1024, low_dim=128,
                 avg_down=False, deep_stem=False,
                 head_type='mlp_head', layer4_dilation=1):
        super(ResNet, self).__init__()
        self.avg_down = avg_down
        self.inplanes = 16 * width
        self.base = int(16 * width)
        self.groups = groups
        self.base_width = width_per_group

        mid_dim = self.base * 8 * block.expansion

        if deep_stem:
            self.conv1 = nn.Sequential(
                conv3x3_bn_relu(in_channel, 32, stride=2),
                conv3x3_bn_relu(32, 32, stride=1),
                conv3x3(32, 64, stride=1)
            )
        else:
            self.conv1 = nn.Conv3d(in_channel, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)

        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base*2, layers[0],stride=2)
        self.layer2 = self._make_layer(block, self.base * 4, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 8, layers[2], stride=2)
        if layer4_dilation == 1:
            self.layer4 = self._make_layer(block, self.base * 16, layers[3], stride=2)
        elif layer4_dilation == 2:
            self.layer4 = self._make_layer(block, self.base * 16, layers[3], stride=1, dilation=2)
        else:
            raise NotImplementedError
        self.avgpool = nn.AvgPool3d(7, stride=1)

        # self.head_type = head_type
        # if head_type == 'mlp_head':
        #     self.fc1 = nn.Linear(mid_dim, mid_dim)
        #     self.relu2 = nn.ReLU(inplace=True)
        #     self.fc2 = nn.Linear(mid_dim, low_dim)
        # elif head_type == 'reduce':
        #     self.fc = nn.Linear(mid_dim, low_dim)
        # elif head_type == 'conv_head':
        #     self.fc1 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
        #     self.bn2 = nn.BatchNorm2d(2048)
        #     self.relu2 = nn.ReLU(inplace=True)
        #     self.fc2 = nn.Linear(mid_dim, low_dim)
        # elif head_type in ['pass', 'early_return', 'multi_layer']:
        #     pass
        # else:
        #     raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # zero gamma for batch norm: reference bag of tricks
        if block is Bottleneck:
            gamma_name = "bn3.weight"
        elif block is BasicBlock:
            gamma_name = "bn2.weight"
        else:
            raise RuntimeError(f"block {block} not supported")
        for name, value in self.named_parameters():
            if name.endswith(gamma_name):
                value.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool3d(kernel_size=stride, stride=stride),
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #c2 = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)



        return [x,c2,c3,c4,c5]


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet18_d(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], deep_stem=True, avg_down=True, **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet34_d(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], deep_stem=True, avg_down=True, **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet50_w2x(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], width=2, **kwargs)


def resnet50_16s(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], layer4_dilation=2, **kwargs)


def resnet50_d(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], deep_stem=True, avg_down=True, **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet101_d(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], deep_stem=True, avg_down=True, **kwargs)


def resnext101_32x8d(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8, **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnet152_d(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], deep_stem=True, avg_down=True, **kwargs)


def resnext152_32x8d(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], groups=32, width_per_group=8, **kwargs)



# 计算模型参数量
def cal_params(mod):
    sum = 0
    for _, param in mod.named_parameters():
        sum += torch.prod(torch.tensor(param.data.size()))
    return sum.item()



class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x

class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class C4_head(nn.Module):
    def __init__(self,in_channel=256,out_channel=512):
        super(C4_head, self).__init__()

        self.conv1 = nn.Conv3d(in_channel,out_channel, kernel_size=(3,3,3), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 2), stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(out_channel, out_channel*2, kernel_size=(2,2,1), stride=1, padding=0, bias=False)

    def forward(self, x, bs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        bs_num,c,w,h,d = x.shape
        x= torch.reshape(x,(bs,bs_num//bs,c*w*h*d))
        return x

class C5_head(nn.Module):
    def __init__(self,in_channel=512,out_channel=1024):
        super(C5_head, self).__init__()


        self.conv1 = nn.Conv3d(in_channel,out_channel, kernel_size=(3,3,2), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, bs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        bs_num, c, w, h, d = x.shape
        x = torch.reshape(x, (bs, bs_num // bs, c * w * h * d))
        return x

class Resnet34(nn.Module):
    def __init__(self, resnet_encoder=None, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(Resnet34, self).__init__()
        self.has_dropout = has_dropout
        self.resnet_encoder = resnet34()

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        self.__init_weight()


    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out

    def forward(self, input, batch_size=4):
        resnet_features = self.resnet_encoder(input)
        out = self.decoder(resnet_features)
        return out

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m,nn.ConvTranspose3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    
    
    
    
    
if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    mod1 = Resnet34()
    print(cal_params(mod1))
    
