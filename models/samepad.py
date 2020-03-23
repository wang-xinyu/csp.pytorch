import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride, dilation=1):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)
        self.dilation = torch.nn.modules.utils._pair(dilation)

    def forward(self, input):
        in_width = input.size()[3]
        in_height = input.size()[2]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))

        effective_kernel_size_width = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        effective_kernel_size_height = (self.kernel_size[1] - 1) * self.dilation[1] + 1

        pad_along_width = ((out_width - 1) * self.stride[0] +
                                   effective_kernel_size_width - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                                    effective_kernel_size_height - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__
