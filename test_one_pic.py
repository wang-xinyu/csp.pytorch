import torch
import cv2
import numpy as np
from models.cspnet import CSPNet_p3p4p5
from utils.keras_weights_loader import load_keras_weights
from utils.utils import *

if __name__ == '__main__':
    device = 'cuda:0'
    img = cv2.imread('shibuye.jpg')
    x = format_img(img)
    #x = np.load('/home/user/wangxinyu/CSP/filename.npy')
    input_dim = x.shape[1:3]
    print(x.shape)
    #print(x)
    #exit(0)
    #x = np.ones([1, 4, 4, 3], dtype='float32')
    x = torch.from_numpy(x).to(device)
    x = x.permute(0, 3, 1, 2)

    model = CSPNet_p3p4p5()
    load_keras_weights(model, 'net_e82_l0.00850005054218.hdf5')
    model.to(device).eval()

    x_cls, x_reg, x_off = model(x)
    Y = [x_cls.detach().cpu().numpy(), x_reg.detach().cpu().numpy(), x_off.detach().cpu().numpy()]
    bboxes = parse_det_offset(Y, input_dim, score=0.4, down=4)

    print('bbox----', bboxes.shape)
    for b in bboxes:
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    cv2.imwrite('out.jpg', img)
