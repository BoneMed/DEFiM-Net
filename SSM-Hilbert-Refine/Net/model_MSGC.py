"""
model_MSGC.py — Multi-Scale Gated Convolution Blocks (self-contained)
=====================================================================
即插即用: 仅依赖 PyTorch, 无需 dynamic_network_architectures / nnunetv2。
支持 Conv1d / Conv2d / Conv3d, 通过 conv_op 参数切换维度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Optional, Type
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


# ======================== Utilities ========================

def _conv_dim(conv_op):
    """从 conv_op 类型推断空间维度 (1/2/3)。"""
    _map = {nn.Conv1d: 1, nn.Conv2d: 2, nn.Conv3d: 3}
    if conv_op in _map:
        return _map[conv_op]
    raise ValueError(f"Unsupported conv op: {conv_op}")


def _to_tuple(conv_op, val):
    """标量 / list / tuple -> 长度匹配空间维度的 tuple。"""
    if isinstance(val, (list, tuple)):
        return tuple(val)
    return (val,) * _conv_dim(conv_op)


# ======================== DropPath ========================

class DropPath(nn.Module):
    """Stochastic Depth: 训练时随机丢弃整条残差分支。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * mask


# ======================== ConvNormAct ========================

class ConvNormAct(nn.Module):
    """Conv -> [Dropout] -> [Norm] -> [Activation]

    各组件为 None 时自动跳过, 行为等价于
    dynamic_network_architectures.ConvDropoutNormReLU。
    """

    def __init__(self,
                 conv_op: Type[_ConvNd],
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]] = 1,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 bias: bool = False,
                 norm_op: Optional[Type[nn.Module]] = None,
                 norm_op_kwargs: Optional[dict] = None,
                 dropout_op: Optional[Type[_DropoutNd]] = None,
                 dropout_op_kwargs: Optional[dict] = None,
                 nonlin: Optional[Type[nn.Module]] = None,
                 nonlin_kwargs: Optional[dict] = None):
        super().__init__()
        ks = _to_tuple(conv_op, kernel_size)
        st = _to_tuple(conv_op, stride)
        pad = tuple((k - 1) // 2 for k in ks)

        layers = [conv_op(in_channels, out_channels, ks,
                          stride=st, padding=pad, bias=bias)]
        if dropout_op is not None:
            layers.append(dropout_op(**(dropout_op_kwargs or {})))
        if norm_op is not None:
            layers.append(norm_op(out_channels, **(norm_op_kwargs or {})))
        if nonlin is not None:
            layers.append(nonlin(**(nonlin_kwargs or {})))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ======================== SpatialNorm ========================

class SpatialNorm(nn.Module):
    """对空间维度 (D, H, W) 做 Layer Normalization。"""

    def __init__(self, input_channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(input_channels))
        self.beta = nn.Parameter(torch.zeros(input_channels))
        self.input_channels = input_channels
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.input_channels,),
                         self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


# ======================== MultiScaleGatedConv ========================

class MultiScaleGatedConv(nn.Module):
    """Multi-scale gated convolution with hierarchical feature aggregation.

    通道层级分解 + 深度可分离卷积 + 逐级聚合。
    支持 1D / 2D / 3D, 通过 conv_op 自动适配。
    """

    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 conv_bias: bool = False,
                 order: int = 5,
                 s: float = 1.0):
        super().__init__()
        self.dim = _conv_dim(conv_op)
        self._scale_factor = s
        self._depth = order

        # Channel splits (ascending order)
        channel_splits = self._compute_channel_hierarchy(input_channels, order)
        self._channel_splits = channel_splits
        total_branch_channels = sum(channel_splits)

        stride = _to_tuple(conv_op, stride)
        ones = (1,) * self.dim
        zeros = (0,) * self.dim

        # Input projection (1x1 conv): input_ch -> input_ch * 2
        self._input_proj = conv_op(
            input_channels, input_channels * 2, ones,
            stride=stride, padding=zeros, dilation=ones, bias=conv_bias
        )

        # Depthwise spatial mixing (kernel=7)
        k7 = (7,) * self.dim
        pad7 = (3,) * self.dim
        self._spatial_mixer = conv_op(
            total_branch_channels, total_branch_channels, k7,
            stride=stride, padding=pad7, dilation=ones,
            bias=conv_bias, groups=total_branch_channels
        )

        # 1x1 conv cascade for progressive refinement
        self._refinement_layers = nn.ModuleList()
        for idx in range(len(channel_splits) - 1):
            self._refinement_layers.append(
                conv_op(channel_splits[idx], channel_splits[idx + 1], ones)
            )

        # Output projection (1x1 conv)
        self._output_proj = conv_op(
            input_channels, output_channels, ones,
            stride=ones, padding=zeros, bias=False
        )

    @staticmethod
    def _compute_channel_hierarchy(base_channels: int, depth: int) -> list:
        """生成各尺度分支的通道数 (升序排列)。"""
        hierarchy = [base_channels >> i for i in range(depth)]
        return hierarchy[::-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 投影到 2x 通道
        projected = self._input_proj(x)

        # 分离: base (最小尺度) + branch (全部尺度)
        base_ch = self._channel_splits[0]
        branch_ch = sum(self._channel_splits)
        base_features, branch_features = projected.split(
            [base_ch, branch_ch], dim=1
        )

        # 深度可分离空间混合 + 缩放
        # PyTorch < 2.1 has no conv_depthwise3d kernel for bf16/fp16, so we
        # force fp32 here when AMP wraps us. The cast back happens at the
        # `mixed_features` consumer.
        prev_dtype = branch_features.dtype
        with torch.cuda.amp.autocast(enabled=False):
            mixed_features = (self._spatial_mixer(branch_features.float())
                              * self._scale_factor)
        mixed_features = mixed_features.to(prev_dtype)

        # 按尺度拆分
        scale_features = mixed_features.split(self._channel_splits, dim=1)

        # 逐级聚合
        aggregated = base_features + scale_features[0]
        for i, refine_layer in enumerate(self._refinement_layers):
            aggregated = refine_layer(aggregated) + scale_features[i + 1]

        return self._output_proj(aggregated)


# ======================== HierarchicalFeatureBlock ========================

class HierarchicalFeatureBlock(nn.Module):
    """Hierarchical feature enhancement module with multi-scale gated convolution.

    即插即用: 可嵌入 encoder / decoder 任意位置作为特征增强块。

    数据流:
        x -> 1x1 降维 -> [门控注意力分支 (带 learnable scale)] -> 残差
          -> Norm -> Conv 变换 -> 残差 -> Activation -> output

    用法示例 (3D):
        block = HierarchicalFeatureBlock(
            conv_op=nn.Conv3d,
            input_channels=128,
            output_channels=128,
            transform_channels=64,
            kernel_size=3,
            nonlin=nn.GELU,
        )
        out = block(x)  # x: [B, 128, D, H, W]
    """

    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 transform_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 conv_bias: bool = False,
                 nonlin: Optional[Type[nn.Module]] = None,
                 nonlin_kwargs: Optional[dict] = None,
                 norm_op: Optional[Type[nn.Module]] = None,
                 norm_op_kwargs: Optional[dict] = None,
                 dropout_op: Optional[Type[_DropoutNd]] = None,
                 dropout_op_kwargs: Optional[dict] = None,
                 layer_eps: float = 0.01,
                 drop_path: float = 0.0,
                 gated_conv_module=MultiScaleGatedConv,
                 order: int = 3):
        super().__init__()
        self._hidden_dim = transform_channels

        ksize = _to_tuple(conv_op, kernel_size)
        st = _to_tuple(conv_op, stride)
        act_kwargs = nonlin_kwargs if nonlin_kwargs is not None else {}

        # ---- Gated attention branch ----
        self._gated_attention = gated_conv_module(
            conv_op, transform_channels, transform_channels, st, conv_bias, order
        )

        # ---- Normalization ----
        self._pre_gate_norm = SpatialNorm(transform_channels, eps=layer_eps)
        self._post_gate_norm = SpatialNorm(transform_channels, eps=layer_eps)

        # ---- 1x1 channel reduction ----
        self._channel_reduce = ConvNormAct(
            conv_op, input_channels, transform_channels, 1, 1, conv_bias
        )

        # ---- Feature transformation path ----
        self._transform_block = nn.Sequential(
            ConvNormAct(
                conv_op, transform_channels, output_channels, ksize, st, conv_bias,
                None, None, dropout_op, dropout_op_kwargs, nonlin, act_kwargs
            ),
            ConvNormAct(
                conv_op, output_channels, output_channels, ksize, st, conv_bias
            )
        )

        # ---- Stochastic depth ----
        self._stochastic_skip = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # ---- Activation ----
        self._activation = nonlin(**act_kwargs) if nonlin is not None else nn.Identity()

        # ---- Learnable gate scale ----
        self._gate_scale = None
        if layer_eps > 0:
            self._gate_scale = nn.Parameter(
                torch.full((transform_channels,), layer_eps), requires_grad=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 1x1 降维
        hidden = self._channel_reduce(x)

        # 门控注意力 + learnable scale
        normalized_hidden = self._pre_gate_norm(hidden)
        gated_features = self._gated_attention(normalized_hidden)
        if self._gate_scale is not None:
            gate_weight = self._gate_scale.reshape(-1, 1, 1, 1)
            gated_features = gate_weight * gated_features
        hidden = hidden + self._stochastic_skip(gated_features)

        # Conv 变换
        normalized_features = self._post_gate_norm(hidden)
        transformed = self._transform_block(normalized_features)

        # 残差融合
        fused = identity + self._stochastic_skip(transformed)
        return self._activation(fused)


