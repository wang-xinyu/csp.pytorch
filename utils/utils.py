import numpy as np
from .py_cpu_nms import py_cpu_nms


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

def parse_det_offset(Y, input_dim, score=0.1,down=4):
    seman = Y[0][0, :, :, 0]
    height = Y[1][0, :, :, 0]
    offset_y = Y[2][0, :, :, 0]
    offset_x = Y[2][0, :, :, 1]
    y_c, x_c = np.where(seman > score)
    print(input_dim)
    print(seman.shape)
    print(y_c.shape)
    print(x_c.shape)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, input_dim[1]), min(y1 + h, input_dim[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = py_cpu_nms(boxs, 0.5)
        boxs = boxs[keep, :]
    return boxs

