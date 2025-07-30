from aiu_fms_testing_utils.utils import (
    sample_sharegpt_requests,
    get_pad_size,
    merge_enforce_keep_hetergenous,
)
from typing import List
from transformers import AutoTokenizer
import pytest
from itertools import product
import os

BATCH_SIZES = [0, 1, 2, 3, 4, 8, 16]
ENFORCE_HETEROGENEOUS = [True, False]
ENFORCE_SIZES = [[], [64, 256], [64, 128, 2048, 4096]]


prompt_max_length = 4096
prompt_min_length = 64
enforce_heterogeneous = True


def expected_error(num_request: int, enforce_sizes: List[int]):
    if num_request < len(enforce_sizes):
        raise ValueError("num request is smaller than enforce_sizes")
    return "OK"


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_merge_enforce_keep_hetergenous(batch_size):
    num_keep = 0
    num_flex = 0
    keep_list = [("keep", 0), ("keep", 2), ("keep", 3)]
    flexible_list = [("flex", 2), ("flex", 3), ("flex", 4), ("flex", 5), ("flex", 6)]
    final_list = merge_enforce_keep_hetergenous(keep_list, flexible_list, batch_size)
    for text, _ in final_list:
        if text == "keep":
            num_keep += 1
        else:
            num_flex += 1
    assert num_keep == len(keep_list)
    if batch_size <= len(keep_list):
        assert num_flex == 0
    else:
        len_unique_num = len(set([item[1] for item in keep_list + flexible_list]))
        assert num_flex == min(batch_size - num_keep, len_unique_num - num_keep)


def test_get_pad_size():
    PAD_MULTIPLE = [64, 128]
    PROMPT_LENGTH = [-65, -1, 0, 1, 63, 64, 65, 128]
    for pad_multiple in PAD_MULTIPLE:
        for prompt_len in PROMPT_LENGTH:
            if prompt_len <= 0:
                assert get_pad_size(prompt_len, pad_multiple) == 0
            elif 1 <= prompt_len <= pad_multiple:
                assert get_pad_size(prompt_len, pad_multiple) == pad_multiple
            elif pad_multiple + 1 <= prompt_len <= 2 * pad_multiple:
                assert get_pad_size(prompt_len, pad_multiple) == 2 * pad_multiple
            else:
                raise NotImplementedError
    # check default 64
    assert get_pad_size(63) == 64


ENFORCE_TEST_COMBO = list(product(BATCH_SIZES, ENFORCE_HETEROGENEOUS, ENFORCE_SIZES))


@pytest.mark.parametrize(
    "batch_size, enforce_heterogeneous, enforce_sizes", ENFORCE_TEST_COMBO
)
def test_enforce_heterogeneous_and_size(
    batch_size, enforce_heterogeneous, enforce_sizes
):
    prompt_max_length = 4096
    prompt_min_length = 64
    dataset = os.environ.get(
        "SHARE_GPT_DATASET_PATH", os.path.expanduser("~/share_gpt.json")
    )
    enforce_size_copy = enforce_sizes.copy()
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-8b-instruct")
    if batch_size < len(enforce_size_copy):
        with pytest.raises(
            ValueError, match="num request is smaller than enforce_sizes"
        ):
            expected_error(batch_size, enforce_size_copy)
    else:
        prompts_and_sizes = sample_sharegpt_requests(
            dataset,
            batch_size,
            tokenizer,
            prompt_min_length,
            prompt_max_length,
            0,
            enforce_heterogeneous,
            enforce_sizes,
        )
        if enforce_size_copy and enforce_heterogeneous:
            # check enforce size
            for size in enforce_size_copy:
                assert size in [get_pad_size(item[1]) for item in prompts_and_sizes]
            # check heterogeneous
            assert len(prompts_and_sizes) == len(
                set(item[1] for item in prompts_and_sizes)
            )
        elif not enforce_size_copy and enforce_heterogeneous:
            # check heterogeneous
            assert len(prompts_and_sizes) == len(
                set(item[1] for item in prompts_and_sizes)
            )
        elif enforce_size_copy and not enforce_heterogeneous:
            # check enforce size
            for size in enforce_size_copy:
                assert size in [get_pad_size(item[1]) for item in prompts_and_sizes]
        # verify the right size is returned
        assert len(prompts_and_sizes) == batch_size
