"""Pruning function implementations for different layer types."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Sequence, Tuple

from .. import ops

__all__=[
    'BasePruningFunc',
    'PrunerBox',

    'prune_conv_out_channels',
    'prune_conv_in_channels',
    'prune_depthwise_conv_out_channels',
    'prune_depthwise_conv_in_channels',
    'prune_batchnorm_out_channels',
    'prune_batchnorm_in_channels',
    'prune_linear_out_channels',
    'prune_linear_in_channels',
    'prune_prelu_out_channels',
    'prune_prelu_in_channels',
    'prune_layernorm_out_channels',
    'prune_layernorm_in_channels',
    'prune_embedding_out_channels',
    'prune_embedding_in_channels',
    'prune_parameter_out_channels',
    'prune_parameter_in_channels',
    'prune_multihead_attention_out_channels',
    'prune_multihead_attention_in_channels',
    'prune_groupnorm_out_channels',
    'prune_groupnorm_in_channels',
    'prune_instancenorm_out_channels',
    'prune_instancenorm_in_channels',
]

class BasePruningFunc(ABC):
    """ Base class for layer pruner.
    It should provide the following functionalities:
        - prune_out_channels: prune out channels of a layer
        - prune_in_channels: prune in channels of a layer
        - get_out_channels: get the number of output channels of a layer
        - get_in_channels: get the number of input channels of a layer
    
    To build the intra-layer dependency, please specify prune_out_channels = prune_in_channels. 

    Example:
    ```python
    class MyPruner(BasePruningFunc):
        def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
            # prune out channels of a layer
            pass
        prune_in_channels = prune_out_channels # this line enables the intra-layer dependency
    ```

    If prune_out_channels != prune_in_channels, there will be no intra-layer dependency.
    """
    TARGET_MODULES = ops.TORCH_OTHERS  # None

    def __init__(self, pruning_dim=1):
        self.pruning_dim = pruning_dim

    @abstractmethod
    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]):
        """Prune output channels of a layer."""
        raise NotImplementedError

    @abstractmethod
    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]):
        """Prune input channels of a layer."""
        raise NotImplementedError

    @abstractmethod
    def get_out_channels(self, layer: nn.Module):
        """Get the number of output channels of a layer."""
        raise NotImplementedError

    @abstractmethod
    def get_in_channels(self, layer: nn.Module):
        """Get the number of input channels of a layer."""
        raise NotImplementedError

    def check(self, layer, idxs, to_output):
        """Validate pruning parameters.
        
        Args:
            layer: The layer to be pruned.
            idxs: List of indices to prune.
            to_output: Whether pruning output channels.
            
        Raises:
            AssertionError: If validation fails.
        """
        pass

    def __call__(self, layer: nn.Module, idxs: Sequence[int], to_output: bool = True, 
                 inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
        """Prune the layer with given indices.
        
        Args:
            layer: The layer to be pruned.
            idxs: List of indices to prune.
            to_output: Whether to prune output channels. Defaults to True.
            inplace: Whether to modify the layer in place. Defaults to True.
            dry_run: Whether this is a dry run (not implemented yet). Defaults to False.
            
        Returns:
            The pruned layer.
        """
        idxs = list(idxs)  # Convert to list to allow sorting
        idxs.sort()
        self.check(layer, idxs, to_output)
        pruning_fn = self.prune_out_channels if to_output else self.prune_in_channels
        if not inplace:
            layer = deepcopy(layer)
        layer = pruning_fn(layer, idxs)
        return layer

    def get_in_channel_groups(self, layer):
        return 1
    
    def get_out_channel_groups(self, layer):
        return 1


class ConvPruner(BasePruningFunc):
    TARGET_MODULE = ops.TORCH_CONV



    def get_out_channels(self, layer):
        return layer.out_channels

    def get_in_channels(self, layer):
        return layer.in_channels

    def get_in_channel_groups(self, layer):
        return layer.groups
    
    def get_out_channel_groups(self, layer):
        return layer.groups


class DepthwiseConvPruner(ConvPruner):
    TARGET_MODULE = ops.TORCH_CONV


    prune_in_channels = prune_out_channels
    # def prune_input(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
    #    return self.prune_output(layer, idxs)


class LinearPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_LINEAR



    def get_out_channels(self, layer):
        return layer.out_features

    def get_in_channels(self, layer):
        return layer.in_features


class BatchnormPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_BATCHNORM


    prune_in_channels = prune_out_channels
    # def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
    #    return self.prune_out_channels(layer=layer, idxs=idxs)

    def get_out_channels(self, layer):
        return layer.num_features

    def get_in_channels(self, layer):
        return layer.num_features


class LayernormPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_LAYERNORM

    def __init__(self, metrcis=None, pruning_dim=-1):
        super().__init__(metrcis)
        self.pruning_dim = pruning_dim



    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.normalized_shape[self.pruning_dim]

    def get_in_channels(self, layer):
        return layer.normalized_shape[self.pruning_dim]

class GroupNormPruner(BasePruningFunc):
    
    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.num_channels

    def get_in_channels(self, layer):
        return layer.num_channels

    def get_in_channel_groups(self, layer):
        return layer.num_groups
    
    def get_out_channel_groups(self, layer):
        return layer.num_groups

class InstanceNormPruner(BasePruningFunc):

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.num_features

    def get_in_channels(self, layer):
        return layer.num_features


class PReLUPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_PRELU


    prune_in_channels = prune_out_channels

    # def prune_in_channels(self, layer:nn.Module, idxs: Sequence[int]) -> nn.Module:
    #    return self.prune_out_channels(layer=layer, idxs=idxs)

    def get_out_channels(self, layer):
        if layer.num_parameters == 1:
            return None
        else:
            return layer.num_parameters

    def get_in_channels(self, layer):
        return self.get_out_channels(layer=layer)

class EmbeddingPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_EMBED


    prune_in_channels = prune_out_channels

    # def prune_in_channels(self, layer: nn.Embedding, idxs: list)-> nn.Module:
    #    return self.prune_out_channels(layer=layer, idxs=idxs)

    def get_out_channels(self, layer):
        return layer.embedding_dim

    def get_in_channels(self, layer):
        return self.get_out_channels(layer=layer)

class LSTMPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_LSTM



    def get_out_channels(self, layer):
        return layer.hidden_size
        
    def get_in_channels(self, layer):
        return layer.input_size
    

class ParameterPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_PARAMETER
    def __init__(self, pruning_dim=-1):
        super().__init__(pruning_dim=pruning_dim)
        

    prune_in_channels = prune_out_channels

    def get_out_channels(self, parameter):
        return parameter.shape[self.pruning_dim]

    def get_in_channels(self, parameter):
        return parameter.shape[self.pruning_dim]


class MultiheadAttentionPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_MHA



    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.embed_dim

    def get_in_channels(self, layer):
        return self.get_out_channels(layer)

PrunerBox = {
    ops.OPTYPE.CONV: ConvPruner(),
    ops.OPTYPE.LINEAR: LinearPruner(),
    ops.OPTYPE.BN: BatchnormPruner(),
    ops.OPTYPE.DEPTHWISE_CONV: DepthwiseConvPruner(),
    ops.OPTYPE.PRELU: PReLUPruner(),
    ops.OPTYPE.LN: LayernormPruner(),
    ops.OPTYPE.EMBED: EmbeddingPruner(),
    ops.OPTYPE.PARAMETER: ParameterPruner(),
    ops.OPTYPE.MHA: MultiheadAttentionPruner(),
    ops.OPTYPE.LSTM: LSTMPruner(),
    ops.OPTYPE.GN: GroupNormPruner(),
    ops.OPTYPE.IN: InstanceNormPruner(),
}

# Alias
prune_conv_out_channels = PrunerBox[ops.OPTYPE.CONV].prune_out_channels
prune_conv_in_channels = PrunerBox[ops.OPTYPE.CONV].prune_in_channels

prune_depthwise_conv_out_channels = PrunerBox[ops.OPTYPE.DEPTHWISE_CONV].prune_out_channels
prune_depthwise_conv_in_channels = PrunerBox[ops.OPTYPE.DEPTHWISE_CONV].prune_in_channels

prune_batchnorm_out_channels = PrunerBox[ops.OPTYPE.BN].prune_out_channels
prune_batchnorm_in_channels = PrunerBox[ops.OPTYPE.BN].prune_in_channels

prune_linear_out_channels = PrunerBox[ops.OPTYPE.LINEAR].prune_out_channels
prune_linear_in_channels = PrunerBox[ops.OPTYPE.LINEAR].prune_in_channels

prune_prelu_out_channels = PrunerBox[ops.OPTYPE.PRELU].prune_out_channels
prune_prelu_in_channels = PrunerBox[ops.OPTYPE.PRELU].prune_in_channels

prune_layernorm_out_channels = PrunerBox[ops.OPTYPE.LN].prune_out_channels
prune_layernorm_in_channels = PrunerBox[ops.OPTYPE.LN].prune_in_channels

prune_embedding_out_channels = PrunerBox[ops.OPTYPE.EMBED].prune_out_channels
prune_embedding_in_channels = PrunerBox[ops.OPTYPE.EMBED].prune_in_channels

prune_parameter_out_channels = PrunerBox[ops.OPTYPE.PARAMETER].prune_out_channels
prune_parameter_in_channels = PrunerBox[ops.OPTYPE.PARAMETER].prune_in_channels

prune_multihead_attention_out_channels = PrunerBox[ops.OPTYPE.MHA].prune_out_channels
prune_multihead_attention_in_channels = PrunerBox[ops.OPTYPE.MHA].prune_in_channels

prune_lstm_out_channels = PrunerBox[ops.OPTYPE.LSTM].prune_out_channels
prune_lstm_in_channels = PrunerBox[ops.OPTYPE.LSTM].prune_in_channels

prune_groupnorm_out_channels = PrunerBox[ops.OPTYPE.GN].prune_out_channels
prune_groupnorm_in_channels = PrunerBox[ops.OPTYPE.GN].prune_in_channels

prune_instancenorm_out_channels = PrunerBox[ops.OPTYPE.IN].prune_out_channels
prune_instancenorm_in_channels = PrunerBox[ops.OPTYPE.IN].prune_in_channels
