import torch
import h5py
import numpy as np
#from cspnet import CSPNet_p3p4p5, ConvBlock

def load_conv_weights(conv, f, layer_name):
    w = np.asarray(f[layer_name][layer_name + '_1/kernel:0'], dtype='float32')
    b = np.asarray(f[layer_name][layer_name + '_1/bias:0'], dtype='float32')
    conv.weight = torch.nn.Parameter(torch.from_numpy(w).permute(3, 2, 0, 1))
    conv.bias = torch.nn.Parameter(torch.from_numpy(b))

def load_bn_weights(bn, f, layer_name):
    w = np.asarray(f[layer_name][layer_name + '_1/gamma:0'], dtype='float32')
    b = np.asarray(f[layer_name][layer_name + '_1/beta:0'], dtype='float32')
    m = np.asarray(f[layer_name][layer_name + '_1/moving_mean:0'], dtype='float32')
    v = np.asarray(f[layer_name][layer_name + '_1/moving_variance:0'], dtype='float32')
    bn.weight = torch.nn.Parameter(torch.from_numpy(w))
    bn.bias = torch.nn.Parameter(torch.from_numpy(b))
    bn.running_mean = torch.from_numpy(m)
    bn.running_var = torch.from_numpy(v)

def load_conv_block_weights(conv_blk, f, blk_name):
    load_conv_weights(conv_blk.conv1, f, 'res' + blk_name + '_branch2a')
    load_bn_weights(conv_blk.bn1, f, 'bn' + blk_name + '_branch2a')
    load_conv_weights(conv_blk.conv2, f, 'res' + blk_name + '_branch2b')
    load_bn_weights(conv_blk.bn2, f, 'bn' + blk_name + '_branch2b')
    load_conv_weights(conv_blk.conv3, f, 'res' + blk_name + '_branch2c')
    load_bn_weights(conv_blk.bn3, f, 'bn' + blk_name + '_branch2c')
    load_conv_weights(conv_blk.conv4, f, 'res' + blk_name + '_branch1')
    load_bn_weights(conv_blk.bn4, f, 'bn' + blk_name + '_branch1')

def load_identity_block_weights(identity_blk, f, blk_name):
    load_conv_weights(identity_blk.conv1, f, 'res' + blk_name + '_branch2a')
    load_bn_weights(identity_blk.bn1, f, 'bn' + blk_name + '_branch2a')
    load_conv_weights(identity_blk.conv2, f, 'res' + blk_name + '_branch2b')
    load_bn_weights(identity_blk.bn2, f, 'bn' + blk_name + '_branch2b')
    load_conv_weights(identity_blk.conv3, f, 'res' + blk_name + '_branch2c')
    load_bn_weights(identity_blk.bn3, f, 'bn' + blk_name + '_branch2c')

def load_l2norm_weights(l2norm, f, layer_name):
    w = np.asarray(f[layer_name][layer_name + '_1/' + layer_name + '_gamma:0'], dtype='float32')
    l2norm.weight = torch.nn.Parameter(torch.from_numpy(w))

def load_keras_weights(model, weights_path):
    with h5py.File(weights_path, 'r') as f:
        print(f.attrs['layer_names'])

        load_conv_weights(model.conv1, f, 'conv1')
        load_bn_weights(model.bn1, f, 'bn_conv1')

        load_conv_block_weights(model.convblk2a, f, '2a')
        load_identity_block_weights(model.identityblk2b, f, '2b')
        load_identity_block_weights(model.identityblk2c, f, '2c')

        load_conv_block_weights(model.convblk3a, f, '3a')
        load_identity_block_weights(model.identityblk3b, f, '3b')
        load_identity_block_weights(model.identityblk3c, f, '3c')
        load_identity_block_weights(model.identityblk3d, f, '3d')

        load_conv_block_weights(model.convblk4a, f, '4a')
        load_identity_block_weights(model.identityblk4b, f, '4b')
        load_identity_block_weights(model.identityblk4c, f, '4c')
        load_identity_block_weights(model.identityblk4d, f, '4d')
        load_identity_block_weights(model.identityblk4e, f, '4e')
        load_identity_block_weights(model.identityblk4f, f, '4f')

        load_conv_block_weights(model.convblk5a, f, '5a')
        load_identity_block_weights(model.identityblk5b, f, '5b')
        load_identity_block_weights(model.identityblk5c, f, '5c')

        load_conv_weights(model.p3, f, 'P3up')
        load_conv_weights(model.p4, f, 'P4up')
        load_conv_weights(model.p5, f, 'P5up')

        load_l2norm_weights(model.p3_l2, f, 'P3norm')
        load_l2norm_weights(model.p4_l2, f, 'P4norm')
        load_l2norm_weights(model.p5_l2, f, 'P5norm')

        load_conv_weights(model.feat, f, 'feat')
        load_bn_weights(model.feat_bn, f, 'bn_feat')

        load_conv_weights(model.center_conv, f, 'center_cls')
        load_conv_weights(model.height_conv, f, 'height_regr')
        load_conv_weights(model.offset_conv, f, 'offset_regr')

