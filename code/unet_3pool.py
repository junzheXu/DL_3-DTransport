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
        self.ec24 = self.encoder2d(384, 512, bias=True, batchnorm=True)
        self.ec25 = self.encoder2d(512, 640, bias=True, batchnorm=True)
        self.ec26 = self.encoder2d(640, 768, bias=True, batchnorm=True)
        self.ec27 = self.encoder2d(768, 896, bias=True, batchnorm=True)

        self.pool20 = nn.MaxPool2d(2)
        self.pool21 = nn.MaxPool2d(2)
        self.pool22 = nn.MaxPool2d(2)

        self.dc12 = self.decoder(896, 896, bias=True)
        self.dc11 = self.encoder2d(640 + 896, 896, bias=True, batchnorm=True)
        self.dc10 = self.encoder2d(896, 512, bias=True, batchnorm=True)
        self.dc9 = self.decoder(512, 512, bias=True)
        self.dc8 = self.encoder2d(384 + 512, 512, bias=True, batchnorm=True)
        self.dc7 = self.encoder2d(512, 256, bias=True, batchnorm=True)
        self.dc6 = self.decoder(256, 256, bias=True)
        self.dc2 = self.encoder2d(128 + 256, 128, kernel_size=1, padding=0, bias=True, batchnorm=False)
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
        syn20 = self.ec20(data_2d)
        e21 = self.ec21(syn20)
        e20 = self.pool20(e21)
        del syn20
        syn21 = self.ec22(e20)
        e23 = self.ec23(syn21)
        e22 = self.pool21(e23)
        del syn21,e20
        syn22 = self.ec24(e22)
        e25 = self.ec25(syn22)
        e24 = self.pool22(e25)
        del syn22
        syn23 = self.ec26(e24)
        e27 = self.ec27(syn23)
        a0 = self.dc12(e27)
        del syn23
        d12 = torch.cat((e25, a0), 1)
        del e25,a0
        d11 = self.dc11(d12)
        del d12
        d10 = self.dc10(d11)
        del d11
        d9 = self.dc9(d10)
        a1 = torch.cat((e23, d9), 1)
        del d9,d10
        d8 = self.dc8(a1)
        d7 = self.dc7(d8)
        del d8,a1
        d6 = self.dc6(d7)
        a3 = torch.cat((e21,d6),1)
        del d6,d7
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
