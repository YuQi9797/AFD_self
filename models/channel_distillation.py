import torch
import torch.nn as nn

from .lsgan import Discriminiator
from .resnet import resnet18, resnet34, resnet50, resnet152
from .wrn import *


def conv1x1_bn(in_channel, out_channel):  # Conv 1x1
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),  # kernel_size = 1, stride = 1, padding = 0
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class ChannelDistillResNet1834(nn.Module):

    def __init__(self, num_classes=1000, dataset_type="imagenet"):
        super().__init__()
        self.student = resnet18(num_classes=num_classes, inter_layer=True, dataset_type=dataset_type)  # not pretrained
        self.teacher = resnet34(num_classes=num_classes, inter_layer=True, dataset_type=dataset_type)  # pretrained

    def forward(self, x):
        ss = self.student(x)  # return: [x1, x2, x3, x4, x]
        ts = self.teacher(x)  # return: [x1, x2, x3, x4, x]
        return ss, ts


class ChannelDistillResNet50152(nn.Module):
    def __init__(self, num_classes=100, dataset_type="imagenet"):
        super().__init__()
        self.student = resnet50(num_classes=num_classes, inter_layer=True,
                                dataset_type=dataset_type)  # 因为是online 所以互相动态学习
        self.teacher = resnet152(num_classes=num_classes, inter_layer=True, dataset_type=dataset_type)

    def forward(self, x):
        # x4: [Batch_size, 2048, H, W]
        ss = self.student(x)  # 若inter_layer = True, 则返回的是列表[x1, x2, x3, x4, x]
        ts = self.teacher(x)  # 同上

        return ss, ts


class DiscriminatorStudentTeacher(nn.Module):
    def __init__(self, in_filters=2048, model_type='res'):
        super().__init__()
        self.discri_s = Discriminiator(in_filters=in_filters, model_type=model_type)
        self.discri_t = Discriminiator(in_filters=in_filters, model_type=model_type)

    def forward(self, x):  # x: [Batch_size, 2048, H, W]
        ss = self.discri_s(x)
        ts = self.discri_t(x)
        return ss, ts


class ChannelDistillWRN1628(nn.Module):  # WRN-16-2  WRN-28-2
    def __init__(self, num_classes=100):
        super().__init__()
        self.student = wrn(num_classes=num_classes, depth=16, widen_factor=2, dropRate=0.0,
                           inter_layer=True)

        self.teacher = wrn(num_classes=num_classes, depth=28, widen_factor=2, dropRate=0.0,
                           inter_layer=True)

    def forward(self, x):
        # x4: [Batch_size, 2048, H, W]
        ss = self.student(x)  # 若inter_layer = True, 则返回的是列表[x1, x2, x3, x4, x]
        ts = self.teacher(x)  # 同上
        return ss, ts
