# from fms.testing.comparison import ModelSignatureParams, compare_model_signatures, get_signature
# from fms.utils import tokenizers
# import pytest
# from fms.models import get_model
# from fms.utils.generation import pad_input_ids
# import itertools
# import torch
# from aiu_fms_testing_utils.testing.validation import extract_validation_information, LogitsExtractorHook, GoldenTokenHook, capture_level_1_metrics, filter_failed_level_1_cases, load_validation_information, validate_level_0, top_k_loss_calculator
# from aiu_fms_testing_utils.utils import warmup_model, sample_sharegpt_requests, ids_for_prompt
# from aiu_fms_testing_utils.utils.aiu_setup import dprint
# import os

# batch_size = 1
# seq_length = 64
# micro_model_kwargs  = {"architecture": "hf_pretrained"}
# model_path_kwargs = {"variant": "deepset/roberta-base-squad2"}

# get_model_kwargs = {**model_path_kwargs, **micro_model_kwargs}

# # prepare the AIU model
# model = get_model(
#     device_type="cpu",
#     fused_weights=False,
#     **get_model_kwargs
# )

# model.eval()
# torch.set_grad_enabled(False)
# model.compile(backend="sendnn")

# # prepare input_ids
# input_ids = torch.randint(3, model.config.src_vocab_size, (batch_size, seq_length), dtype=torch.int64)
# position_ids = torch.arange(0, seq_length).repeat(batch_size, 1)
# is_pad = input_ids == -1
# mask = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)

# # warmup model
# logits_getter_fn = lambda x: x if isinstance(x, torch.Tensor) else torch.cat(list(x), dim=-1)
# other_params = {"mask": mask, "position_ids": position_ids}
# aiu_msp = ModelSignatureParams(model, ["x"], logits_getter_fn=logits_getter_fn, inp=input_ids, other_params=other_params)
# get_signature(aiu_msp.model, aiu_msp.params, aiu_msp.inp, aiu_msp.other_params, aiu_msp.logits_getter_fn)

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}
res = nlp(QA_input)