'''
This opcounter is adapted from https://github.com/sovrasov/flops-counter.pytorch

Copyright (C) 2021 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import numpy as np
import torch.nn as nn
import torch

try:
    import timm
    has_timm = True
except:
    has_timm = False

@torch.no_grad()
def count_ops_and_params(model, example_inputs, layer_wise=False):
    global CUSTOM_MODULES_MAPPING
    ori_model = model 
    model = copy.deepcopy(model) # deepcopy to avoid changing the original model
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count(ost=sys.stdout, verbose=False,
                                ignore_list=[])
    if isinstance(example_inputs, (tuple, list)):
        _ = flops_model(*example_inputs)
    elif isinstance(example_inputs, dict):
        _ = flops_model(**example_inputs)
    else:
        _ = flops_model(example_inputs)
    flops_count, params_count, _layer_flops, _layer_params = flops_model.compute_average_flops_cost()
    layer_flops = {}
    layer_params = {}
    for ori_m, m in zip(ori_model.modules(), model.modules()):
        layer_flops[ori_m] = _layer_flops.get(m)
        layer_params[ori_m] = _layer_params.get(m)

    flops_model.stop_flops_count()
    CUSTOM_MODULES_MAPPING = {}
    if layer_wise:
        return flops_count, params_count, layer_flops, layer_params
    return flops_count, params_count

















def rnn_flops_counter_hook(rnn_module, input, output):
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    If sigmoid and tanh are hard, only a comparison FLOPS should be accurate
    """
    pass








CUSTOM_MODULES_MAPPING = {}

MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_flops_counter_hook,
    nn.Conv2d: conv_flops_counter_hook,
    nn.Conv3d: conv_flops_counter_hook,
    # activations
    nn.ReLU: relu_flops_counter_hook,
    nn.PReLU: relu_flops_counter_hook,
    nn.ELU: relu_flops_counter_hook,
    nn.LeakyReLU: relu_flops_counter_hook,
    nn.ReLU6: relu_flops_counter_hook,
    # poolings
    nn.MaxPool1d: pool_flops_counter_hook,
    nn.AvgPool1d: pool_flops_counter_hook,
    nn.AvgPool2d: pool_flops_counter_hook,
    nn.MaxPool2d: pool_flops_counter_hook,
    nn.MaxPool3d: pool_flops_counter_hook,
    nn.AvgPool3d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_flops_counter_hook,
    # BNs
    nn.BatchNorm1d: bn_flops_counter_hook,
    nn.BatchNorm2d: bn_flops_counter_hook,
    nn.BatchNorm3d: bn_flops_counter_hook,

    nn.InstanceNorm1d: bn_flops_counter_hook,
    nn.InstanceNorm2d: bn_flops_counter_hook,
    nn.InstanceNorm3d: bn_flops_counter_hook,
    nn.GroupNorm: bn_flops_counter_hook,
    nn.LayerNorm: ln_flops_counter_hook,
    # FC
    nn.Linear: linear_flops_counter_hook,
    # Upscale
    nn.Upsample: upsample_flops_counter_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_flops_counter_hook,
    nn.ConvTranspose2d: conv_flops_counter_hook,
    nn.ConvTranspose3d: conv_flops_counter_hook,
    # RNN
    nn.RNN: rnn_flops_counter_hook,
    nn.GRU: rnn_flops_counter_hook,
    nn.LSTM: rnn_flops_counter_hook,
    nn.RNNCell: rnn_cell_flops_counter_hook,
    nn.LSTMCell: rnn_cell_flops_counter_hook,
    nn.GRUCell: rnn_cell_flops_counter_hook,
    nn.MultiheadAttention: multihead_attention_counter_hook
}

if has_timm:
    MODULES_MAPPING.update(
        {
            timm.models.vision_transformer.Attention: timm_multihead_attention_counter_hook,
            timm.models.swin_transformer.WindowAttention: timm_multihead_attention_counter_hook,
        }
    )

if hasattr(nn, 'GELU'):
    MODULES_MAPPING[nn.GELU] = relu_flops_counter_hook


import sys
from functools import partial
import torch.nn as nn
import copy

def accumulate_flops(self, layer_flops):
    if is_supported_instance(self):
        layer_flops[self] = self.__flops__
        return self.__flops__
    else:
        sum = 0
        for m in self.children():
            sum += m.accumulate_flops(layer_flops)
        layer_flops[self] = sum
        return sum


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters())
    return params_num


def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(
                                                    net_main_module)

    net_main_module.reset_flops_count()

    return net_main_module

def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Returns current mean flops consumption per image.
    """

    for m in self.modules():
        m.accumulate_flops = accumulate_flops.__get__(m)

    layer_flops = {}
    flops_sum = self.accumulate_flops(layer_flops)

    for m in self.modules():
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    layer_params = {}
    for m in self.modules():
        layer_params[m] = get_model_parameters_number(m)

    params_sum = get_model_parameters_number(self)
    return flops_sum / self.__batch_counter__, params_sum, layer_flops, layer_params


def start_flops_count(self, **kwargs):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Activates the computation of mean flops consumption per image.
    Call it before you run the network.
    """
    add_batch_counter_hook_function(self)

    seen_types = set()


    self.apply(partial(add_flops_counter_hook_function, **kwargs))


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.
    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)
    self.apply(remove_flops_counter_variables)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Resets statistics computed so far.
    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


# ---- Internal functions


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__




def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False




