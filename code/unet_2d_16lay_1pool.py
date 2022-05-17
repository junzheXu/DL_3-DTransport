# -*- coding: utf-8 -*-
"""
U-NET model.
"J.Z. Xu; H. R. Zhang, Z. Cheng, J. Y. Liu, Y. Y. Xu and Y. C. Wang. Approximating Three-dimensional (3-D) Transport of Atmospheric Pollutants via Deep Learning"
"""
import torch
import torch.nn as nn
import time


class UNet(nn.Module):
    def __init__(self, in_channel2, out_channel):
        self.in_channel2 = in_channel2
        self.out_channel = out_channel
        super(UNet, self).__init__()

        # 2d
        self.ec20 = self.encoder2d(self.in_channel2, 80,bias=True, batchnorm=True)
        self.ec21 = self.encoder2d(80, 128,bias=True, batchnorm=True)
        self.ec22 = self.encoder2d(128, 256, bias=True, batchnorm=True)
        self.ec23 = self.encoder2d(256, 384, bias=True, batchnorm=True)

        self.pool20 = nn.MaxPool2d(2)

        self.dc6 = self.decoder(384, 384, bias=True)
        self.dc2 = self.encoder2d(128 + 384, 128, kernel_size=1, padding=0, bias=True, batchnorm=False)
        self.dc1 = self.encoder2d_last(128, 64, kernel_size=1, padding=0, bias=True, batchnorm=False)
        self.dc0 = self.encoder2d_last(64, out_channel, kernel_size=1, padding=0, bias=True, batchnorm=False)

    def encoder2d(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                  bias=True, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer
    def encoder2d_last(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                  bias=True, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels))
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, dilation=1, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer

    def forward(self, data_2d):
        # 2d
        syn20 = self.ec20(data_2d)
        e21 = self.ec21(syn20)
        e20 = self.pool20(e21)
        del syn20
        syn21 = self.ec22(e20)
        e23 = self.ec23(syn21)
        d6 = self.dc6(e23)
        a3 = torch.cat((e21,d6),1)
        del d6,e23
        d2 = self.dc2(a3)
        del a3
        d1 = self.dc1(d2)
        d0 = self.dc0(d1)
        return d0

    def load(self, path):

        self.load_state_dict(torch.load(path, map_location='cuda'))

    def save(self, name=None, vision=None, epoch=None):
        if name is None:
            prefix = vision + '_' + 'epoch%s' % (epoch)
            name = time.strftime(
                '/lustre/home/acct-esehazenet/hazenet-pg5/python_scripts/train_gpu/checkpoints/' + '%m%d_%H:%M:%S' + prefix + '.pth')
        torch.save(self.module.state_dict(), name)
        return name
