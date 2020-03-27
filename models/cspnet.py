import torch
import math
import torch.nn as nn
from .l2norm import L2Norm
from .samepad import SamePad2d

class IdentityBlock(nn.Module):
    expansion = 4
    def __init__(self, inchannels, filters, dila=1):
        super(IdentityBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, filters, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(filters, eps=1e-03, momentum=0.01)
        self.samepad = SamePad2d(3, 1, dilation=dila)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, dilation=dila)
        self.bn2 = nn.BatchNorm2d(filters, eps=1e-03, momentum=0.01)
        self.conv3 = nn.Conv2d(filters, filters * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(filters * self.expansion, eps=1e-03, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        print('a shape --- ', out.shape)
        out = self.samepad(out)
        print('b shape --- ', out.shape)
        out = self.conv2(out)
        print('c shape --- ', out.shape)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += x
        out = self.relu(out)

        return out

class ConvBlock(nn.Module):
    expansion = 4
    def __init__(self, inchannels, filters, s=2, dila=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, filters, kernel_size=1, stride=s)
        self.bn1 = nn.BatchNorm2d(filters, eps=1e-03, momentum=0.01)
        self.samepad = SamePad2d(3, 1, dilation=dila)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, dilation=dila)
        self.bn2 = nn.BatchNorm2d(filters, eps=1e-03, momentum=0.01)
        self.conv3 = nn.Conv2d(filters, filters * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(filters * self.expansion, eps=1e-03, momentum=0.01)
        self.conv4 = nn.Conv2d(inchannels, filters * self.expansion, kernel_size=1, stride=s)
        self.bn4 = nn.BatchNorm2d(filters * self.expansion, eps=1e-03, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        print('a shape --- ', out.shape)
        out = self.samepad(out)
        print('b shape --- ', out.shape)
        out = self.conv2(out)
        print('c shape --- ', out.shape)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = self.conv4(x)
        shortcut = self.bn4(shortcut)
        print('shortcut shape --- ', shortcut.shape)

        out += shortcut
        out = self.relu(out)

        return out


class CSPNet_p3p4p5(nn.Module):
    def __init__(self, num_scale=1):
        super(CSPNet_p3p4p5, self).__init__()

        #resnet = resnet50(pretrained=True, receptive_keep=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-03, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.samepad1 = SamePad2d(3, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.convblk2a = ConvBlock(64, 64, s=1)
        self.identityblk2b = IdentityBlock(256, 64)
        self.identityblk2c = IdentityBlock(256, 64)

        self.convblk3a = ConvBlock(256, 128)
        self.identityblk3b = IdentityBlock(512, 128)
        self.identityblk3c = IdentityBlock(512, 128)
        self.identityblk3d = IdentityBlock(512, 128)

        self.convblk4a = ConvBlock(512, 256)
        self.identityblk4b = IdentityBlock(1024, 256)
        self.identityblk4c = IdentityBlock(1024, 256)
        self.identityblk4d = IdentityBlock(1024, 256)
        self.identityblk4e = IdentityBlock(1024, 256)
        self.identityblk4f = IdentityBlock(1024, 256)

        self.convblk5a = ConvBlock(1024, 512, s=1, dila=2)
        self.identityblk5b = IdentityBlock(2048, 512, dila=2)
        self.identityblk5c = IdentityBlock(2048, 512, dila=2)

        self.p3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.p4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=4, padding=0)
        self.p5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=4, padding=0)

        nn.init.xavier_normal_(self.p3.weight)
        nn.init.xavier_normal_(self.p4.weight)
        nn.init.xavier_normal_(self.p5.weight)
        nn.init.constant_(self.p3.bias, 0)
        nn.init.constant_(self.p4.bias, 0)
        nn.init.constant_(self.p5.bias, 0)

        self.p3_l2 = L2Norm(256, 10)
        self.p4_l2 = L2Norm(256, 10)
        self.p5_l2 = L2Norm(256, 10)

        self.feat = nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1)
        self.feat_bn = nn.BatchNorm2d(256, eps=1e-03, momentum=0.01)

        self.center_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.height_conv = nn.Conv2d(256, num_scale, kernel_size=1)
        self.offset_conv = nn.Conv2d(256, 2, kernel_size=1)

        nn.init.xavier_normal_(self.feat.weight)
        nn.init.xavier_normal_(self.center_conv.weight)
        nn.init.xavier_normal_(self.height_conv.weight)
        nn.init.xavier_normal_(self.offset_conv.weight)
        nn.init.constant_(self.center_conv.bias, -math.log(0.99/0.01))
        nn.init.constant_(self.height_conv.bias, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.samepad1(x)
        x = self.maxpool(x)

        x = self.convblk2a(x)
        x = self.identityblk2b(x)
        stage2 = self.identityblk2c(x)

        x = self.convblk3a(stage2)
        x = self.identityblk3b(x)
        x = self.identityblk3c(x)
        stage3 = self.identityblk3d(x)

        x = self.convblk4a(stage3)
        x = self.identityblk4b(x)
        x = self.identityblk4c(x)
        x = self.identityblk4d(x)
        x = self.identityblk4e(x)
        stage4 = self.identityblk4f(x)

        x = self.convblk5a(stage4)
        x = self.identityblk5b(x)
        stage5 = self.identityblk5c(x)

        p3up = self.p3(stage3)
        p4up = self.p4(stage4)
        p5up = self.p5(stage5)
        p3up = self.p3_l2(p3up)
        p4up = self.p4_l2(p4up)
        p5up = self.p5_l2(p5up)
        cat = torch.cat([p3up, p4up, p5up], dim=1)

        feat = self.feat(cat)
        feat = self.feat_bn(feat)
        feat = self.relu(feat)

        x_cls = self.center_conv(feat)
        x_cls = torch.sigmoid(x_cls)
        x_reg = self.height_conv(feat)
        x_off = self.offset_conv(feat)

        x_cls = x_cls.permute(0, 2, 3, 1)
        x_reg = x_reg.permute(0, 2, 3, 1)
        x_off = x_off.permute(0, 2, 3, 1)

        return x_cls, x_reg, x_off

    # def train(self, mode=True):
    #     # Override train so that the training mode is set as we want
    #     nn.Module.train(self, mode)
    #     if mode:
    #         # Set fixed blocks to be in eval mode
    #         self.conv1.eval()
    #         self.layer1.eval()
    #
    #         # bn is trainable in CONV2
    #         def set_bn_train(m):
    #             class_name = m.__class__.__name__
    #             if class_name.find('BatchNorm') != -1:
    #                 m.train()
    #             else:
    #                 m.eval()
    #         self.layer1.apply(set_bn_train)

