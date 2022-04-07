import math

import torch.nn as nn

from .DCNv2.dcn_v2 import DCN


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class PoseResNet(nn.Module):
    def __init__(self, backbone, heads, t):
        super(PoseResNet, self).__init__()
        self.heads = heads
        self.backbone = backbone
        # 反卷积
        self.deconv_layer = self.make_deconv_layer(t)
        # 根据task建立相应的输出层
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(64, 64,
                          kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, classes,
                          kernel_size=1, stride=1,
                          padding=0, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def make_deconv_layer(self, t):
        layers = []
        if '18' in t or '34' in t:
            i = 1
        else:
            i = 4
        inplanes = 512 * i
        num_filters = [256, 128, 64]
        for i in range(3):
            # fc层
            fc = DCN(inplanes, num_filters[i],
                     kernel_size=(3, 3), stride=1, padding=1,
                     dilation=1, deformable_groups=1)
            # bn层
            bn = nn.BatchNorm2d(num_filters[i])
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
            # 反卷积层
            up = nn.ConvTranspose2d(in_channels=num_filters[i],
                                    out_channels=num_filters[i],
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    bias=False)
            # 反卷积参数初始化
            fill_up_weights(up)

            layers.append(fc)
            layers.append(bn)
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(bn)
            layers.append(nn.ReLU(inplace=True))
            inplanes = num_filters[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv_layer(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

