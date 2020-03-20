import torch
import cv2
import numpy as np
from cspnet import CSPNet_p3p4p5
from keras_weights_loader import load_keras_weights


img_channel_mean = [103.939, 116.779, 123.68]

def format_img_channels(img):
    """ formats the image channels based on config """
    # img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= img_channel_mean[0]
    img[:, :, 1] -= img_channel_mean[1]
    img[:, :, 2] -= img_channel_mean[2]
    # img /= C.img_scaling_factor
    # img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def format_img(img):
    """ formats an image for model prediction based on config """
    img = format_img_channels(img)
    return img #return img, ratio


if __name__ == '__main__':
    device = 'cuda:0'
    #img = cv2.imread('/home/user/wangxinyu/CSP/data/caltech/test/images/set06_V000_I00299.jpg')
    #x = format_img(img)
    x = np.load('/home/user/wangxinyu/CSP/filename.npy')
    print(x.shape)
    #print(x)
    #exit(0)
    #x = np.ones([1, 4, 4, 3], dtype='float32')
    x = torch.from_numpy(x).to(device)
    x = x.permute(0, 3, 1, 2)

    model = CSPNet_p3p4p5()
    load_keras_weights(model, 'temp.hdf5')
    model.to(device).eval()

    x_cls, x_reg, x_off = model(x)

    print('cls----', x_cls.shape)
    print(x_cls)
    print('reg----', x_reg.shape)
    print(x_reg)
    print('off----', x_off.shape)
    print(x_off)
