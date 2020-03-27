import torch
import cv2
import numpy as np
from models.cspnet import CSPNet_p3p4p5
from utils.keras_weights_loader import load_keras_weights
from utils.utils import *

device = 'cuda:1'

def detect_face(model, img, scale=1, flip=False):
    img_h, img_w = img.shape[:2]
    img_h_new, img_w_new = int(np.ceil(scale * img_h / 16) * 16), int(np.ceil(scale * img_w / 16) * 16)
    scale_h, scale_w = img_h_new / img_h, img_w_new / img_w

    img_s = cv2.resize(img, None, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)
    # img_h, img_w = img_s.shape[:2]
    # print frame_number
    input_dim = [img_h_new, img_w_new]

    if flip:
        img_sf = cv2.flip(img_s, 1)
        # x_rcnn = format_img_pad(img_sf, C)
        x_rcnn = format_img(img_sf)
    else:
        # x_rcnn = format_img_pad(img_s, C)
        x_rcnn = format_img(img_s)
    x = torch.from_numpy(x_rcnn).to(device)
    x = x.permute(0, 3, 1, 2)
    x_cls, x_reg, x_off = model(x)
    print('x reg shape ', x_reg.shape)
    Y = [x_cls.detach().cpu().numpy(), x_reg.detach().cpu().numpy(), x_off.detach().cpu().numpy()]
    boxes = parse_wider_offset(Y, input_dim, score=0.3, nmsthre=0.4)
    if len(boxes) > 0:
        keep_index = np.where(np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) >= 12)[0]
        boxes = boxes[keep_index, :]
    if len(boxes) > 0:
        if flip:
            boxes[:, [0, 2]] = img_s.shape[1] - boxes[:, [2, 0]]
        boxes[:, 0:4:2] = boxes[:, 0:4:2] / scale_w
        boxes[:, 1:4:2] = boxes[:, 1:4:2] / scale_h
    else:
        boxes = np.empty(shape=[0, 5], dtype=np.float32)
    return boxes



if __name__ == '__main__':

    model = CSPNet_p3p4p5(num_scale=2)
    load_keras_weights(model, 'net_e382_l0.hdf5')
    model.to(device).eval()

    img = cv2.imread('worlds-largest-selfie.jpg')
    bboxes = detect_face(model, img)

    print('bbox----', bboxes.shape)
    for b in bboxes:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
    cv2.imwrite('out.jpg', img)
