import random
import time
from typing import Any, Callable, List, MutableMapping, Optional, Tuple, Union, Unpack

from fms.modules.attention import AttentionKwargs, _sdpa_compute_op, register_attention_op
import torch
import torch.nn as nn

from torch.library import custom_op
import torch.nn.functional as F

@custom_op("aiu::paged_attn_store", mutates_args=(), device_types="cpu")
def paged_attn_store(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    result_key_cache = key_cache.clone()
    result_value_cache = value_cache.clone()
    for seq_i, slot_mapping_seq in enumerate(slot_mapping):
        for tok_i, slot in enumerate(slot_mapping_seq):
            block_number = slot.item() // 64
            position = slot.item() % 64

            result_key_cache[block_number, position, :, :] = key[seq_i, tok_i, :, :]
            result_value_cache[block_number, position, :, :] = value[seq_i, tok_i, :, :]
    return result_key_cache, result_value_cache

@paged_attn_store.register_fake
def paged_attn_store_meta(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return key_cache, value_cache

def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


@custom_op("aiu::paged_attn_compute", mutates_args={}, device_types="cpu")
def paged_attn_compute(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float,
    current_tkv_mask: torch.Tensor,
    left_padded_prompt_mask: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    # torch.zeros(NUM_BLOCKS, BLOCK_SIZE, kvheads, head_size, dtype=model_dtype),
    output = torch.zeros_like(query)
    num_query_heads = query.shape[2]
    num_kv_heads = value_cache.shape[2]
    head_size = value_cache.shape[3]
    block_size = value_cache.shape[1]
    num_seqs = query.shape[0]

    block_tables_lst = block_table.cpu().tolist()

    seq_lens_lst = current_tkv_mask.cpu().tolist()
    for i in range(num_seqs):
        q = query[i]
        block_table = block_tables_lst[i]
        start_pos = left_padded_prompt_mask[i].item()
        seq_len = int(seq_lens_lst[i])

        keys_lst: list[torch.Tensor] = []
        values_lst: list[torch.Tensor] = []
        for j in range(start_pos, seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, block_offset, :, :]
            k = k.reshape(num_kv_heads, head_size)
            keys_lst.append(k)

            v = value_cache[block_number, block_offset, :, :]
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_kv_heads > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_query_heads // num_kv_heads, dim=1)
            values = torch.repeat_interleave(
                values, num_query_heads // num_kv_heads, dim=1
            )

        out = ref_masked_attention(q, keys, values, scale)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)
    return output


@paged_attn_compute.register_fake
def paged_attn_compute_meta(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float,
    current_tkv_mask: torch.Tensor,
    left_padded_prompt_mask: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros_like(query)

class AIUPagedAttentionKwargs(AttentionKwargs):
    current_tkv_mask: Optional[torch.Tensor]
    left_padded_prompt_mask: Optional[torch.Tensor]
    block_table: Optional[torch.Tensor]
    slot_mapping: torch.Tensor
    mask: Optional[torch.Tensor] # prefill mask

def __aiu_paged_store_op(
    keys: torch.Tensor,
    values: torch.Tensor,
    key_cache: Optional[torch.Tensor],
    value_cache: Optional[torch.Tensor],
    **attn_kwargs: Unpack[AIUPagedAttentionKwargs],
):
    result_key_cache, result_value_cache = torch.ops.aiu.paged_attn_store(
        keys, values, key_cache, value_cache, attn_kwargs["slot_mapping"]
    )

    # for prefill, we want to return the original keys/values
    if attn_kwargs.get("block_table", None) is None:
        return keys, values, result_key_cache, result_value_cache
    else:
        return result_key_cache, result_value_cache, result_key_cache, result_value_cache


def __aiu_paged_compute_op(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    nheads: int,
    kvheads: int,
    p_dropout: float,
    scale_factor: float,
    **attn_kwargs: Unpack[AIUPagedAttentionKwargs],
):
    return torch.ops.aiu.paged_attn_compute(
        query,
        key_cache,
        value_cache,
        scale_factor,
        attn_kwargs["current_tkv_mask"],
        attn_kwargs["left_padded_prompt_mask"],
        attn_kwargs["block_table"]
    )

register_attention_op(
    "aiu_paged_attn",
    __aiu_paged_store_op,
    _sdpa_compute_op,
    is_prefill_op=lambda **attn_kwargs: attn_kwargs.get("block_table", None) is None,
    compute_decode_op=__aiu_paged_compute_op,
)

class AIUPagedModelWrapper(nn.Module):

    def __init__(self, model: nn.Module, num_blocks: int = 100, block_size: int = 64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.config = model.config
        if hasattr(model, "head"):
            self.dtype = model.head.weight.dtype
        elif hasattr(model, "shared"):
            self.dtype = model.shared.head.weight.dtype
        else:
            self.dtype = torch.float32
        
        self.nheads = model.config.nheads
        if hasattr(model.config, "kvheads"):
            self.kvheads = model.config.kvheads
        elif hasattr(model.config, "multiquery_attn"):
            self.kvheads = 1 if model.config.multiquery_attn else model.config.nheads
        else:
            self.kvheads = self.nheads

        tensor_parallel_size = (
            model.distributed_strategy.group.size()
            if hasattr(model.distributed_strategy, "group")
            else 1
        )
        self.kvheads = self.kvheads // tensor_parallel_size if self.kvheads > 1 else self.kvheads
        self.head_size = model.config.emb_dim // self.nheads
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.current_tkv_mask = None
        self.left_padded_prompt_mask = None
        self._reset_blocks()

    def _reset_blocks(self):
        self.block_numbers = [i for i in range(self.num_blocks)]
        random.seed(0)
        random.shuffle(self.block_numbers)

    def _prepare_prefill_kwargs(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> AIUPagedAttentionKwargs:
        self.position_i = input_ids.size(1) - 1
        self.left_padded_prompt_mask = (position_ids == 0).sum(dim=1) - 1
        current_context_lengths = (position_ids != 0).sum(dim=1) + 1
        self.current_tkv_mask = self.left_padded_prompt_mask + current_context_lengths
        slot_mapping = []
        block_table = []
        for seq_i in input_ids:
            block_table_i = []
            slot_mapping_i = []
            for pos_i in range(seq_i.size(0)):
                if pos_i % self.block_size == 0:
                    block_number = self.block_numbers.pop(0)
                    block_table_i.append(block_number)
                block_offset = pos_i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping_i.append(slot)
            slot_mapping.append(slot_mapping_i)
            block_table.append(block_table_i)

        self.block_table = block_table
        return AIUPagedAttentionKwargs(
            attn_name="aiu_paged_attn", 
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int64)
        )

    def _run_prefill(self, x, position_ids, only_last_token, mask):
        self._reset_blocks()
        attn_kwargs: AIUPagedAttentionKwargs = self._prepare_prefill_kwargs(x, position_ids)
        past_key_value_states = [
            (
                torch.zeros(self.num_blocks, self.block_size, self.kvheads, self.head_size, dtype=self.dtype),
                torch.zeros(self.num_blocks, self.block_size, self.kvheads, self.head_size, dtype=self.dtype),
            )
            for _ in range(self.model.config.nlayers)
        ]

        mask = mask.unsqueeze(1)
        
        outputs_list = []
        for seq_i in range(x.size(0)):
            input_ids_i = x[seq_i].unsqueeze(0)
            slot_mapping_i = attn_kwargs["slot_mapping"][seq_i].unsqueeze(0)
            position_ids_i = position_ids[seq_i].unsqueeze(0)
            mask_i = mask[seq_i].unsqueeze(0)

            # batch dynamic
            torch._dynamo.mark_static(input_ids_i, 0)
            torch._dynamo.mark_static(slot_mapping_i, 0)
            torch._dynamo.mark_static(position_ids_i, 0)
            torch._dynamo.mark_static(mask_i, 0)

            # seq dynamic
            torch._dynamo.mark_dynamic(input_ids_i, 1)
            torch._dynamo.mark_dynamic(slot_mapping_i, 1)
            torch._dynamo.mark_dynamic(position_ids_i, 1)
            torch._dynamo.mark_dynamic(mask_i, 2)
            torch._dynamo.mark_dynamic(mask_i, 3)

            attn_kwargs_i: AIUPagedAttentionKwargs = AIUPagedAttentionKwargs(
                attn_name="aiu_paged_attn", 
                slot_mapping=slot_mapping_i,
                mask=mask_i
            )

            output, past_key_value_states = self.model(input_ids_i, position_ids=position_ids_i, past_key_value_states=past_key_value_states, use_cache=True, only_last_token=only_last_token, **attn_kwargs_i)
            
            outputs_list.append(output[0].squeeze(0))
        
        return (torch.stack(outputs_list), past_key_value_states)
    
    def _run_decode(self, x, position_ids, only_last_token, past_key_value_states):
        self.position_i = self.position_i + 1
        if self.position_i % self.block_size == 0:
            for block_table_i in self.block_table:
                block_number = self.block_numbers.pop(0)
                block_table_i.append(block_number)
        block_offset = self.position_i % self.block_size

        slot_mapping = []
        for block_table_i in self.block_table:
            slot = block_table_i[-1] * self.block_size + block_offset
            slot_mapping.append([slot])
        block_table_tensor = torch.tensor(self.block_table, dtype=torch.int64)
        slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int64)
        self.current_tkv_mask = self.current_tkv_mask + 1
        # batch
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(block_table_tensor, 0)
        torch._dynamo.mark_dynamic(slot_mapping_tensor, 0)
        torch._dynamo.mark_dynamic(position_ids, 0)
        torch._dynamo.mark_dynamic(self.current_tkv_mask, 0)
        torch._dynamo.mark_dynamic(self.left_padded_prompt_mask, 0)

        # seq
        torch._dynamo.mark_static(x, 1)  # always 1
        torch._dynamo.mark_dynamic(block_table_tensor, 1)
        torch._dynamo.mark_static(slot_mapping_tensor, 1)  # always 1
        torch._dynamo.mark_static(position_ids, 1)  # always 1

        attn_kwargs: AIUPagedAttentionKwargs = AIUPagedAttentionKwargs(
            attn_name="aiu_paged_attn", 
            current_tkv_mask=self.current_tkv_mask, 
            left_padded_prompt_mask=self.left_padded_prompt_mask, 
            block_table=block_table_tensor, 
            slot_mapping=slot_mapping_tensor
        )

        return self.model(x, position_ids=position_ids, only_last_token=only_last_token, past_key_value_states=past_key_value_states, use_cache=True, **attn_kwargs)

    def forward(
        self,
        x: torch.Tensor, 
        position_ids: torch.LongTensor, 
        past_key_value_states: Optional[Tuple[torch.FloatTensor,]] = None,
        only_last_token: bool = False,
        mask: Optional[torch.Tensor] = None,
        **kwargs # we create AIUPagedAttentionKwargs internally here, so no need to pass
    ):
        
        if past_key_value_states is None:
            return self._run_prefill(x, position_ids, only_last_token, mask)
        else:
            return self._run_decode(x, position_ids, only_last_token, past_key_value_states)


def __update_padding_kwargs(
    use_cache: bool, model_specific_kwargs: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Generic function to prepare any model specific keyword arguments"""
    # extend the attention mask
    mask = model_specific_kwargs.get("mask", None)
    if mask is not None:
        # get the last row of the 3d mask
        mask = mask[:, -1:, :]
        # extend the mask one slot
        mask = torch.cat(
            (
                mask,
                torch.zeros(mask.size(0), 1, 1, device=mask.device),
            ),
            dim=2,
        )
        model_specific_kwargs["mask"] = mask

    # extend the position_ids
    position_ids = model_specific_kwargs.get("position_ids", None)
    if position_ids is not None:
        if use_cache:
            position_ids = position_ids[:, -1:] + 1
        else:
            position_ids = torch.cat(
                (position_ids, position_ids[:, -1:] + 1),
                dim=1,
            )
        model_specific_kwargs["position_ids"] = position_ids
    return model_specific_kwargs

def generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: torch.Tensor,
    max_seq_len: int = 4096,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 10,
    do_sample: bool = True,
    num_beams: int = 1,
    use_cache: bool = False,
    contiguous_cache: bool = False,
    eos_token_id: Optional[int] = None,
    timing: str = "",
    post_iteration_hook: Optional[
        Callable[
            [int, torch.Tensor, torch.Tensor, MutableMapping[str, Any]],
            Tuple[torch.Tensor, MutableMapping[str, Any]],
        ]
    ] = None,
    extra_kwargs: Optional[MutableMapping[str, Any]] = None,
):
    """
    A trivial generate function that can be used for validation/testing in
    cases where HF is not available.
    We could add implementations for other types of generation, but this is
    enough for making sure a model is working.
    Does not implement beam search, but this can be added.

    Args:
        model: A function or nn.Module that takes a batch of input_ids and
            returns logits
        input_ids: a rectangular tensor of input_ids (batch x seq)
        max_seq_len: the sequence length of the model
        max_new_tokens: max tokens to generate
        temperature: temperature of softmax when sampling
        top_k: only search among top k tokens
        do_sample: multinomial sampling. False for greedy.
        num_beams: TODO: support beam search
        use_cache: requires that the model accept use_cache and
            past_key_value_states args in forward method.
        contiguous_cache: ensures the cache is contiguous in device memory
        eos_token_id: the optional token id representing the end of sequence
        timing: whether to measure timings: "per-token" for each token generation time,
            "e2e" for full generation loop. Both options make `generate` return a tuple
            with the following information:
            - "per-token": Array with `max_new_tokens` time measurements (in s)
            - "e2e": Array with a single e2e generation loop time measurement (in s)
        post_iteration_hook: a function that will get called after each iteration.
            It must have the following signature: f(int token_position, Tensor logits, Tensor next_val, Dict kwargs) ->
            Tuple[Tensor next_val, Dict kwargs]. If it is defined, will replace next_val
            and kwargs based on the contents of the function.
        extra_kwargs: an optional mapping of additional kwargs to pass to the model.
            For example: if extra_kwargs contains position_ids and mask keys, these
            model parameters will be updated as-appropriate for each token generated.
    """
    if num_beams != 1:
        raise NotImplementedError("generate() does yet not support beam search")

    kwargs: MutableMapping[str, Any] = dict()
    if extra_kwargs is not None:
        kwargs.update(extra_kwargs)

    if isinstance(input_ids, torch.Tensor):
        is_batch = len(input_ids.shape) > 1
        # our model requires batch dimension
        if not is_batch:
            input_ids = input_ids.unsqueeze(0)
    else:
        raise TypeError("input_ids must be one of Tensor or List")

    eos_found = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=input_ids.device
    )

    result = input_ids
    next_input = input_ids
    NUM_BLOCKS = 100
    BLOCK_SIZE = 64
    if hasattr(model, "head"):
        model_dtype = model.head.weight.dtype
    elif hasattr(model, "shared"):
        model_dtype = model.shared.head.weight.dtype
    else:
        model_dtype = torch.float32

    nheads = model.config.nheads
    if hasattr(model.config, "kvheads"):
        kvheads = model.config.kvheads
    elif hasattr(model.config, "multiquery_attn"):
        kvheads = 1 if model.config.multiquery_attn else model.config.nheads
    else:
        kvheads = nheads

    tensor_parallel_size = (
        model.distributed_strategy.group.size()
        if hasattr(model.distributed_strategy, "group")
        else 1
    )
    kvheads = kvheads // tensor_parallel_size if kvheads > 1 else kvheads
    head_size = model.config.emb_dim // nheads
    kwargs["attn_name"] = "aiu_paged_attn"
    kwargs["past_key_value_states"] = [
        (
            torch.zeros(NUM_BLOCKS, BLOCK_SIZE, kvheads, head_size, dtype=model_dtype),
            torch.zeros(NUM_BLOCKS, BLOCK_SIZE, kvheads, head_size, dtype=model_dtype),
        )
        for _ in range(model.config.nlayers)
    ]
    kwargs["block_table"] = None
    block_numbers = [i for i in range(NUM_BLOCKS)]
    random.seed(0)
    random.shuffle(block_numbers)
    left_padded_prompt_mask = (kwargs["position_ids"] == 0).sum(dim=1) - 1
    current_context_lengths = (kwargs["position_ids"] != 0).sum(dim=1) + 1
    current_tkv_mask = left_padded_prompt_mask + current_context_lengths
    slot_mapping = []
    block_table = []
    for seq_i in input_ids:
        block_table_i = []
        slot_mapping_i = []
        for pos_i in range(seq_i.size(0)):
            if pos_i % BLOCK_SIZE == 0:
                block_number = block_numbers.pop(0)
                block_table_i.append(block_number)
            block_offset = pos_i % BLOCK_SIZE
            slot = block_number * BLOCK_SIZE + block_offset
            slot_mapping_i.append(slot)
        slot_mapping.append(slot_mapping_i)
        block_table.append(block_table_i)
    kwargs["slot_mapping"] = torch.tensor(slot_mapping, dtype=torch.int64)
    kwargs["current_tkv_mask"] = None
    kwargs["left_padded_prompt_mask"] = None
    kwargs["use_cache"] = use_cache

    prompt_length = input_ids.shape[1]

    if timing != "":
        times: List[float] = []
        start_time = time.time()

    for i in range(max_new_tokens):
        input_ids = next_input[:, -max_seq_len:]

        # prepare any padding keyword arguments
        # iteration 0 is the prefill step (cache has not been filled yet), so no need to extend the mask/position_ids
        if i > 0:
            kwargs["mask"] = None
            kwargs = __update_padding_kwargs(use_cache, kwargs)
            pos_i = result.size(1) - 1
            if pos_i % BLOCK_SIZE == 0:
                for block_table_i in block_table:
                    block_number = block_numbers.pop(0)
                    block_table_i.append(block_number)
            block_offset = pos_i % BLOCK_SIZE

            slot_mapping = []
            for block_table_i in block_table:
                slot = block_table_i[-1] * BLOCK_SIZE + block_offset
                slot_mapping.append([slot])
            kwargs["block_table"] = torch.tensor(block_table, dtype=torch.int64)
            kwargs["slot_mapping"] = torch.tensor(slot_mapping, dtype=torch.int64)
            current_tkv_mask = current_tkv_mask + 1
            kwargs["current_tkv_mask"] = current_tkv_mask
            kwargs["left_padded_prompt_mask"] = left_padded_prompt_mask

        # prefill
        if i == 0:
            kwargs["mask"] = kwargs["mask"].unsqueeze(1)
            
            outputs_list = []
            current_kv_cache = kwargs["past_key_value_states"]
            for seq_i in range(input_ids.size(0)):
                input_ids_i = input_ids[seq_i].unsqueeze(0)
                slot_mapping_i = kwargs["slot_mapping"][seq_i].unsqueeze(0)
                position_ids_i = kwargs["position_ids"][seq_i].unsqueeze(0)
                mask_i = kwargs["mask"][seq_i].unsqueeze(0)

                # batch dynamic
                torch._dynamo.mark_static(input_ids_i, 0)
                torch._dynamo.mark_static(slot_mapping_i, 0)
                torch._dynamo.mark_static(position_ids_i, 0)
                torch._dynamo.mark_static(mask_i, 0)

                # seq dynamic
                torch._dynamo.mark_dynamic(input_ids_i, 1)
                torch._dynamo.mark_dynamic(slot_mapping_i, 1)
                torch._dynamo.mark_dynamic(position_ids_i, 1)
                torch._dynamo.mark_dynamic(mask_i, 2)
                torch._dynamo.mark_dynamic(mask_i, 3)

                only_last_token = kwargs.get("only_last_token", False)

                output, current_kv_cache = model(input_ids_i, slot_mapping=slot_mapping_i, position_ids=position_ids_i, mask=mask_i, past_key_value_states=current_kv_cache, use_cache=kwargs["use_cache"], only_last_token=only_last_token, attn_name=kwargs["attn_name"])
                
                outputs_list.append(output[0].squeeze(0))
            
            output = (torch.stack(outputs_list), current_kv_cache)

        # decode
        else:
            # mask is no longer used here

            # batch
            torch._dynamo.mark_dynamic(input_ids, 0)
            torch._dynamo.mark_dynamic(kwargs["block_table"], 0)
            torch._dynamo.mark_dynamic(kwargs["slot_mapping"], 0)
            torch._dynamo.mark_dynamic(kwargs["position_ids"], 0)
            torch._dynamo.mark_dynamic(kwargs["current_tkv_mask"], 0)
            torch._dynamo.mark_dynamic(kwargs["left_padded_prompt_mask"], 0)

            # seq
            torch._dynamo.mark_static(input_ids, 1)  # always 1
            torch._dynamo.mark_dynamic(kwargs["block_table"], 1)
            torch._dynamo.mark_static(kwargs["slot_mapping"], 1)  # always 1
            torch._dynamo.mark_static(kwargs["position_ids"], 1)  # always 1

            output = model(input_ids, **kwargs)
        if use_cache:
            logits, past_key_value_states = output
            # TODO: this should go away when reduce-overhead issues are fixed, or
            # maybe could be moved into model code to be more portable.
            kwargs["past_key_value_states"] = past_key_value_states
        else:
            logits = output

        if "only_last_token" not in kwargs:
            logits = logits[:, -1, :]

        if do_sample:
            # get logits from last value in sequence nad scale
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_val = torch.multinomial(probs, num_samples=1)
        else:
            next_val = torch.argmax(logits, dim=-1).unsqueeze(0).t()

        if post_iteration_hook is not None:
            next_val, kwargs = post_iteration_hook(
                i + prompt_length, logits, next_val, kwargs
            )

        result = torch.cat((result, next_val), dim=-1)

        # avoid continuing to generate if all have reached EOS
        if eos_token_id is not None:
            eos_found = torch.logical_or(eos_found, next_val == eos_token_id)
            if torch.sum(eos_found) == input_ids.shape[0]:
                break

        if use_cache:
            next_input = next_val
        else:
            next_input = result

        if timing == "per-token":
            if input_ids.device.type == "cuda":
                torch.cuda.synchronize()
            current_token_time = time.time() - start_time
            times.append(current_token_time)
            start_time = time.time()

    if timing == "e2e":
        if input_ids.device.type == "cuda":
            torch.cuda.synchronize()
        e2e_time = time.time() - start_time
        times.append(e2e_time)

    if not is_batch:
        result = result[0]

    if timing != "":
        return result, times
    return result