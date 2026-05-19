import torch
import torch.nn as nn
import torch.nn.functional as F

def _extract_patches(x, kernel_size, stride, padding):
    """
    :param x: The input feature maps.  (batch_size, in_c, h, w)
    :param kernel_size: the kernel size of the conv filter (tuple of two elements)
    :param stride: the stride of conv operation  (tuple of two elements)
    :param padding: number of paddings. be a tuple of two elements
    :return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    pass





class ComputeMatGrad:

    @classmethod
    def __call__(cls, input, grad_output, layer):
        if isinstance(layer, nn.Linear):
            grad = cls.linear(input, grad_output, layer)
        elif isinstance(layer, nn.Conv2d):
            grad = cls.conv2d(input, grad_output, layer)
        else:
            raise NotImplementedError
        return grad

    @staticmethod
    def linear(input, grad_output, layer):
        """
        :param input: batch_size * input_dim
        :param grad_output: batch_size * output_dim
        :param layer: [nn.module] output_dim * input_dim
        :return: batch_size * output_dim * (input_dim + [1 if with bias])
        """
        pass

    @staticmethod
    def conv2d(input, grad_output, layer):
        """
        :param input: batch_size * in_c * in_h * in_w
        :param grad_output: batch_size * out_c * h * w
        :param layer: nn.module batch_size * out_c * (in_c*k_h*k_w + [1 if with bias])
        :return:
        """
        pass
