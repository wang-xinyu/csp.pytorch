import torch
import cv2
import numpy as np
import os
from models.cspnet import CSPNet_p3p4p5
from utils.keras_weights_loader import load_keras_weights
from utils.utils import *

if __name__ == '__main__':
    device = 'cuda:0'
    weights_path = 'net_e121_l0.hdf5'
    out_path = 'output/valresults/city/h/off/121'
    input_dim = [1024, 2048]

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    res_file = os.path.join(out_path, 'val_det.txt')

    model = CSPNet_p3p4p5()
    load_keras_weights(model, weights_path)
    model.to(device).eval()

    f = open('data/citypersons/val.txt', 'r')
    files = f.readlines();
    num_imgs = len(files)

    res_all = []
    for i in range(0, num_imgs):
        l = files[i]
        print(l)

        img = cv2.imread('data/citypersons/leftImg8bit/val/' + l.strip())
        x = format_img(img)
        with torch.no_grad():
            x = torch.from_numpy(x).to(device)
            x = x.permute(0, 3, 1, 2)
            x_cls, x_reg, x_off = model(x)
        Y = [x_cls.detach().cpu().numpy(), x_reg.detach().cpu().numpy(), x_off.detach().cpu().numpy()]
        boxes = parse_det_offset(Y, input_dim, score=0.1, down=4)

        if len(boxes)>0:
            f_res = np.repeat(i + 1, len(boxes), axis=0).reshape((-1, 1))
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]
            res_all += np.concatenate((f_res, boxes), axis=-1).tolist()
    np.savetxt(res_file, np.array(res_all), fmt='%6f')

    f.close()
    exit(0)

