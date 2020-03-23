import torch
import cv2
import numpy as np
import os
from models.cspnet import CSPNet_p3p4p5
from utils.keras_weights_loader import load_keras_weights
from utils.utils import *

if __name__ == '__main__':
    device = 'cuda:0'
    weights_path = 'net_e82_l0.00850005054218.hdf5'
    out_path = 'output/valresults/caltech/h/off/82'
    input_dim = [480, 640]

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for st in range(6, 11):
        set_path = os.path.join(out_path, 'set' + '%02d' % st)
        if not os.path.exists(set_path):
            os.mkdir(set_path)

    model = CSPNet_p3p4p5()
    load_keras_weights(model, weights_path)
    model.to(device).eval()

    f = open('data/caltech/test.txt', 'r')
    files = f.readlines();
    num_imgs = len(files)

    for i in range(0, num_imgs):
        l = files[i]
        print(l)
        st = l.split('_')[0]
        video = l.split('_')[1]

        frame_number = int(l.split('_')[2][1:6]) + 1
        frame_number_next = int(files[i + 1].split('_')[2][1:6]) + 1 if i < num_imgs - 1 else -1
        print('next', frame_number_next)
        set_path = os.path.join(out_path, st)
        video_path = os.path.join(set_path, video + '.txt')

        if os.path.exists(video_path):
            continue
        if frame_number == 30:
            res_all = []

        img = cv2.imread('data/caltech/images/' + l.strip())
        x = format_img(img)
        x = torch.from_numpy(x).to(device)
        x = x.permute(0, 3, 1, 2)
        x_cls, x_reg, x_off = model(x)
        Y = [x_cls.detach().cpu().numpy(), x_reg.detach().cpu().numpy(), x_off.detach().cpu().numpy()]
        boxes = parse_det_offset(Y, input_dim, score=0.01, down=4)

        if len(boxes)>0:
            f_res = np.repeat(frame_number, len(boxes), axis=0).reshape((-1, 1))
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]
            res_all += np.concatenate((f_res, boxes), axis=-1).tolist()
        if frame_number_next == 30 or i == num_imgs - 1:
            np.savetxt(video_path, np.array(res_all), fmt='%6f')

    f.close()
    exit(0)

