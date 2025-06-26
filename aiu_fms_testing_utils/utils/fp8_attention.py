import functools
import math
from importlib.util import find_spec
from typing import (
    NotRequired,
    Optional,
    Tuple,
    Unpack,
)
import torch
from torch import Tensor

from fms.modules.attention import (
    AttentionKwargs,
    register_attention_op,
    _sdpa_update_attn_kwargs,
)


########
## scaled_bmm - A batched version of _scaled_mm
########
@torch.library.custom_op("spyre::scaled_bmm", mutates_args=())
def spyre_scaled_bmm(
    mat1: Tensor,
    mat2: Tensor,
    scale1: Tensor,
    scale2: Tensor,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
) -> Tensor:
    assert mat1.shape[:-2] == mat2.shape[:-2], (
        "batch dimensions must match for mat1 and mat2"
    )
    mat1 = mat1.view(-1, *mat1.shape[-2:])
    mat2 = mat2.view(-1, *mat2.shape[-2:])
    out = torch.empty(
        (mat1.shape[0], mat1.shape[1], mat2.shape[2]),
        dtype=out_dtype,
        device=mat1.device,
    )
    for b_idx in range(mat1.shape[0]):
        out[b_idx] = torch._scaled_mm(
            mat1[b_idx],
            mat2[b_idx],
            scale1,
            scale2,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )
    return out.view(*mat1.shape[:-2], mat1.shape[1], mat2.shape[2])


# All that's needed for torch.compile support
@spyre_scaled_bmm.register_fake
def _(
    mat1: Tensor,
    mat2: Tensor,
    scale1: Tensor,
    scale2: Tensor,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
) -> Tensor:
    return torch.empty(
        (*mat1.shape[:-2], mat1.shape[-2], mat2.shape[-1]),
        dtype=out_dtype,
        device=mat1.device,
    )

_HANDLED_FUNCTIONS = {}


def _implements(torch_function):
    """Register a torch function override"""
    def decorator(func):
        @functools.wraps(torch_function)
        def wrapper(f, types, args, kwargs):
            return func(f, types, args, kwargs)

        _HANDLED_FUNCTIONS[torch_function] = wrapper
        return func

    return decorator



class ScaledTensor(torch.Tensor):
    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=data.dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )

    def __init__(
            self,
            data: torch.Tensor,
            scale: torch.Tensor,
    ):
        self._data = data
        self._scale = scale

    def __tensor_flatten__(self):
        ctx = {}
        return ["_data", "_scale"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        assert len(inner_tensors) == 2
        return ScaledTensor(
            inner_tensors["_data"],
            inner_tensors["_scale"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func in _HANDLED_FUNCTIONS:
            return _HANDLED_FUNCTIONS[func](func, types, args, kwargs)

        arg_types = tuple(type(arg) for arg in args)
        kwarg_types = {k: type(arg) for k, arg in kwargs.items()}
        raise NotImplementedError(
            f"{cls.__name__} dispatch: attempting to run unimplemented operator/function: {func=}, {types=}, {arg_types=}, {kwarg_types=}"
        )

    def __repr__(self):
        return f"{self._data.__repr__()}\n{self._scale.__repr__()}"


# TODO: Figure out better scales for AIU? These come from vLLM
Q_RANGE = 200.0
K_RANGE = 200.0
V_RANGE = 100.0


class MathFP8AttentionKwargs(AttentionKwargs):
    mask: NotRequired[Tensor]
    do_scale_q: bool
    is_causal_mask: bool


def _construct_fp8_cache(
    tensor: torch.Tensor, scale: torch.Tensor
) -> ScaledTensor:
    # Construct the torchao tensor to save kv cache with its scales
    return ScaledTensor(
        tensor,
        scale,
    )


def _math_fp8_store_op(
    keys: torch.Tensor,
    values: torch.Tensor,
    key_cache: Optional[torch.Tensor],
    value_cache: Optional[torch.Tensor],
    **attn_kwargs: Unpack[MathFP8AttentionKwargs],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(key_cache, ScaledTensor) and isinstance(
        value_cache, ScaledTensor
    ):
        k_scale = key_cache._scale
        v_scale = value_cache._scale
    else:
        k_scale = (torch.abs(keys).max() / K_RANGE).to(dtype=torch.float32)
        v_scale = (torch.abs(values).max() / V_RANGE).to(dtype=torch.float32)

    keys = (keys / k_scale).to(torch.float8_e4m3fn).transpose(2, 1)
    values = (values / v_scale).to(torch.float8_e4m3fn).transpose(2, 1)

    if (
        isinstance(key_cache, ScaledTensor)
        and isinstance(value_cache, ScaledTensor)
        and value_cache.numel() > 0
    ):
        key_cache = torch.cat((key_cache._data, keys), dim=2)
        value_cache = torch.cat((value_cache._data, values), dim=2)
        key_cache = _construct_fp8_cache(key_cache, k_scale)
        value_cache = _construct_fp8_cache(value_cache, v_scale)
        return (
            key_cache,
            value_cache,
            key_cache,
            value_cache,
        )
    else:
        keys = _construct_fp8_cache(keys.contiguous(), k_scale)
        values = _construct_fp8_cache(values.contiguous(), v_scale)
        return (keys, values, keys, values)


def _math_fp8_compute_op(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    nheads: int,
    kvheads: int,
    p_dropout: float,
    scale_factor: Optional[float],
    **attn_kwargs: Unpack[MathFP8AttentionKwargs],
) -> torch.Tensor:
    orig_dtype = query.dtype

    q_scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)
    if attn_kwargs.get("do_scale_q", False):
        q_scale.copy_(torch.abs(query).max() / Q_RANGE)
        query = query / q_scale

    query = query.to(torch.float8_e4m3fn).transpose(2, 1)

    if (
        isinstance(key_cache, ScaledTensor) and 
        isinstance(value_cache, ScaledTensor)
    ):
        k_scale = key_cache._scale
        v_scale = value_cache._scale
        key_cache = key_cache._data
        value_cache = value_cache._data
    else:
        k_scale = (torch.abs(key_cache).max() / K_RANGE).to(dtype=torch.float32)
        v_scale = (torch.abs(value_cache).max() / V_RANGE).to(dtype=torch.float32)
        key_cache = (key_cache / k_scale).to(torch.float8_e4m3fn)
        value_cache = (value_cache / v_scale).to(torch.float8_e4m3fn)

    # no longer transposing prior to store, so need to check this in case of no cache
    # TODO: Refactor FMS to avoid edge cases where this fails by adding use_cache param here
    if key_cache.shape[1] != kvheads and key_cache.shape[2] == kvheads:
        key_cache = key_cache.transpose(2, 1)
        value_cache = value_cache.transpose(2, 1)

    mask = attn_kwargs.get("mask", None)
    if mask is not None:
        # Our expected mask format is bs x q_len x k_len, so to make it broadcastable
        # we need to create the nheads dimension
        while len(mask.size()) != 4:  # expects bs (x nheads) x q_len x kv_len
            mask = mask.unsqueeze(1)

    L, S = query.size(-2), key_cache.size(-2)
    scale_factor = (
        1 / math.sqrt(query.size(-1)) if scale_factor is None else scale_factor
    )
    attn_bias = torch.zeros(L, S, dtype=orig_dtype, device=query.device)
    if attn_kwargs.get("is_causal_mask", False):
        assert mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(torch.float32)

    if mask is not None:
        if mask.dtype == torch.bool:
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
        else:
            attn_bias = mask + attn_bias

    expansion = nheads // kvheads
    if expansion > 1:
        key_cache = key_cache.repeat_interleave(
            query.size(-3) // key_cache.size(-3), -3
        )
        value_cache = value_cache.repeat_interleave(
            query.size(-3) // value_cache.size(-3), -3
        )

    attn_weight = (
        torch.ops.spyre.scaled_bmm(
            query,
            key_cache.transpose(-2, -1),
            q_scale,
            k_scale,
            out_dtype=orig_dtype,
            use_fast_accum=True,
        )
        * scale_factor
    )
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, p_dropout, train=True)
    # Do matmul in orig_dtype
    attn = attn_weight @ (value_cache.to(dtype=orig_dtype) * v_scale)

    attn = attn.to(orig_dtype).transpose(2, 1).contiguous()
    return attn


register_attention_op(
    "math_fp8",
    _math_fp8_store_op,
    _math_fp8_compute_op,
    update_attn_kwargs_op=_sdpa_update_attn_kwargs,
)