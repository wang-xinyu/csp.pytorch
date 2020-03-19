import torch
import math
import torch.nn as nn
import h5py
import numpy as np
#from resnet import *
from l2norm import L2Norm


class CSPNet_p3p4p5(nn.Module):
    def __init__(self):
        super(CSPNet_p3p4p5, self).__init__()

        #resnet = resnet50(pretrained=True, receptive_keep=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        #self.bn1 = resnet.bn1
        #self.relu = resnet.relu
        #self.maxpool = resnet.maxpool
        #self.layer1 = resnet.layer1
        #self.layer2 = resnet.layer2
        #self.layer3 = resnet.layer3
        #self.layer4 = resnet.layer4

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

        self.feat = nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_bn = nn.BatchNorm2d(256, momentum=0.01)
        self.feat_act = nn.ReLU(inplace=True)

        self.pos_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.reg_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.off_conv = nn.Conv2d(256, 2, kernel_size=1)

        nn.init.xavier_normal_(self.feat.weight)
        nn.init.xavier_normal_(self.pos_conv.weight)
        nn.init.xavier_normal_(self.reg_conv.weight)
        nn.init.xavier_normal_(self.off_conv.weight)
        nn.init.constant_(self.pos_conv.bias, -math.log(0.99/0.01))
        nn.init.constant_(self.reg_conv.bias, 0)
        nn.init.constant_(self.off_conv.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        return x
        #x = self.bn1(x)
        #x = self.relu(x)
        #x = self.maxpool(x)

        #x = self.layer1(x)

        #x = self.layer2(x)
        #p3 = self.p3(x)
        #p3 = self.p3_l2(p3)

        #x = self.layer3(x)
        #p4 = self.p4(x)
        #p4 = self.p4_l2(p4)

        #x = self.layer4(x)
        #p5 = self.p5(x)
        #p5 = self.p5_l2(p5)

        #cat = torch.cat([p3, p4, p5], dim=1)

        #feat = self.feat(cat)
        #feat = self.feat_bn(feat)
        #feat = self.feat_act(feat)

        #x_cls = self.pos_conv(feat)
        #x_cls = torch.sigmoid(x_cls)
        #x_reg = self.reg_conv(feat)
        #x_off = self.off_conv(feat)

        #return x_cls, x_reg, x_off

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
    def load_keras_weights(self, weights_path):
        with h5py.File(weights_path, 'r') as f:
            #model_weights = f['model_weights']
            #layer_names = list(map(str, model_weights.keys()))
            #state_dict = OrderedDict()
            
            print(f.attrs['layer_names'])
            print(f['conv1'].attrs.keys())
            print(f['conv1'].attrs['weight_names'])
            print(f['conv1']['conv1_1/kernel:0'])

            w = np.asarray(f['conv1']['conv1_1/kernel:0'], dtype='float32')
            b = np.asarray(f['conv1']['conv1_1/bias:0'], dtype='float32')
            self.conv1.weight = torch.nn.Parameter(torch.from_numpy(w).permute(3, 2, 0, 1))
            self.conv1.bias = torch.nn.Parameter(torch.from_numpy(b))
            print('b shape ', b.shape)


            print(self.conv1.weight.shape)
            print(self.conv1.bias.shape)
            #print('weight, ', self.conv1.weight.permute(2, 3, 1, 0))
            #print('bias, ', self.conv1.bias)
            #num_w = conv_layer.weight.numel()
            #conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            #conv_layer.weight.data.copy_(conv_w)




