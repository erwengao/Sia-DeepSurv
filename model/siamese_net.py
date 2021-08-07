import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F


class siaRes(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1):
        super().__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size//2, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, )
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, 256)
        self.fc1 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout3d(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        # if isinstance(out.data, torch.cuda.FloatTensor):
        #     zero_pads = zero_pads.cuda()
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:  # basic expansion为1 bottle为4  输入与输出不等使用短路，否则直接相加
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(
            block(in_planes=self.in_planes,      # 输出为planes*expansion
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion  # 将输出改为下次输入
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))  # 输入输出相等（planes为参数）输出为planes*expansion
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.conv1(x)   # 尺寸减小为4分之一，卷积减一半，pooling减一半
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)   # 64channel 尺寸不变
        x = self.layer2(x)   # 128channel 尺寸减小一半 有一个卷积层步长为2
        x = self.layer3(x)   # 256channel 尺寸减小一半 有一个卷积层步长为2
        x = self.layer4(x)   # 512channel 尺寸减小一半 有一个卷积层步长为2
        x = self.avgpool(x)
        output_fea = x.view(x.size(0), -1)
        output_fea = self.dropout(output_fea)
        output_fea = self.fc(output_fea)
        output = self.dropout(output_fea)
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.sigmoid(output)
        return output_fea, output

    def forward(self, input_a, input_b):
        output_a1, output_a = self.forward_once(input_a)
        output_b1, output_b = self.forward_once(input_b)
        output = torch.cat((output_a1, output_b1), dim=1)
        return output_a, output_b, output


##3d
def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


def generate_resmodel(model_depth, pretrain=True, **kwargs):
    global model
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    if model_depth == 10:
        model = siaRes(BasicBlock3d, [1, 1, 1, 1], get_inplanes(), **kwargs)
        if pretrain:
            w = torch.load('/data/gaohan/deeplearning/MedicalNet_pytorch_files2/pretrain/resnet_10_23dataset.pth', map_location='cpu')
            model.load_state_dict(w, strict=False)
    elif model_depth == 18:
        model = siaRes(BasicBlock3d, [2, 2, 2, 2], get_inplanes(), **kwargs)
        if pretrain:
            w = torch.load('/data/gaohan/deeplearning/MedicalNet_pytorch_files2/pretrain/resnet_18_23dataset.pth', map_location='cpu')
            model.load_state_dict(w, strict=False)
    elif model_depth == 34:
        model = siaRes(BasicBlock3d, [3, 4, 6, 3], get_inplanes(), **kwargs)
        if pretrain:
            w = torch.load('/data/gaohan/deeplearning/MedicalNet_pytorch_files2/pretrain/resnet_34_23dataset.pth', map_location='cpu')
            model.load_state_dict(w, strict=False)
    elif model_depth == 50:
        model = siaRes(Bottleneck3d, [3, 4, 6, 3], get_inplanes(), **kwargs)
        if pretrain:
            w = torch.load('/data/gaohan/deeplearning/MedicalNet_pytorch_files2/pretrain/resnet_50_23dataset.pth', map_location='cpu')
            model.load_state_dict(w, strict=False)
    elif model_depth == 101:
        model = siaRes(Bottleneck3d, [3, 4, 23, 3], get_inplanes(), **kwargs)
        if pretrain:
            w = torch.load('/data/gaohan/deeplearning/MedicalNet_pytorch_files2/pretrain/resnet_101.pth', map_location='cpu')
            model.load_state_dict(w, strict=False)
    elif model_depth == 152:
        model = siaRes(Bottleneck3d, [3, 8, 36, 3], get_inplanes(), **kwargs)
        if pretrain:
            w = torch.load('/data/gaohan/deeplearning/MedicalNet_pytorch_files2/pretrain/resnet_152.pth', map_location='cpu')
            model.load_state_dict(w, strict=False)
    elif model_depth == 200:
        model = siaRes(Bottleneck3d, [3, 24, 36, 3], get_inplanes(), **kwargs)
        if pretrain:
            w = torch.load('/data/gaohan/deeplearning/MedicalNet_pytorch_files2/pretrain/resnet_200.pth', map_location='cpu')
            model.load_state_dict(w, strict=False)
    return model


class BasicBlock3d(nn.Module):
    expansion = 1  # 输出维度是输入维度的一倍

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
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


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
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

# class siamese(nn.Module):
#     def __init__(self):
#         super(siamese, self).__init__()
#         self.conv1 = nn.Conv3d(3, 64, 3, padding=1)
#         self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
#         self.linear1 = nn.Linear(128 * 10 * 16 * 16, 512)
#         self.linear2 = nn.Linear(512, 1)
#         self.relu = nn.ReLU(inplace=True)
#         self.pooling = nn.MaxPool3d(3, stride=(1, 2, 2), padding=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward_once(self, x):
#         output = self.conv1(x)
#         output = self.pooling(output)
#         output1 = self.relu(output)
#         output_1 = output1.view(output1.size(0), -1)
#         output2 = self.conv2(output1)
#         output2 = self.pooling(output2)
#         output2 = self.relu(output2)
#         output_2 = output2.view(output2.size(0), -1)
#         output_512 = self.linear1(output_2)
#         output = self.linear2(output_512)
#         output = self.sigmoid(output)
#         # return output_1, output_2, output
#         return output_512, output
#
#     def forward(self, input_a, input_b):
#         # output_a1, output_a2, output_a = self.forward_once(input_a)
#         # output_b1, output_b2, output_b = self.forward_once(input_b)
#         # cat1 = torch.cat((output_a1, output_b1), dim=1)
#         # cat2 = torch.cat((output_a2, output_b2), dim=1)
#         # output = torch.cat((cat1, cat2), dim=1)
#
#         output_a1, output_a = self.forward_once(input_a)
#         output_b1, output_b = self.forward_once(input_b)
#         output = torch.cat((output_a1, output_b1), dim=1)
#
#         return output_a, output_b, output