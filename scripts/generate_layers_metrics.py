import os
import time
import logging
import argparse

import itertools
import torch
import torch.nn as nn

from fms.utils import tokenizers
from fms.models import get_model
from fms.utils.generation import pad_input_ids, generate

from aiu_fms_testing_utils.testing.validation import get_default_validation_prefix

from aiu_fms_testing_utils.utils import (
    sample_sharegpt_requests,
    ids_for_prompt,
)


logger = logging.getLogger(__name__)
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(message)s")

parser = argparse.ArgumentParser(
    description="Script to generate the model's metrics by layer"
)

parser.add_argument(
    "--mode",
    choices=["generate", "model-forward"],
    default="generate",
    required=True,
    help="Sets the output generation mode."
)

args = parser.parse_args()
mode = args.mode

ORIGINAL_HF_HOME = os.environ.get("HF_HOME", None)

SHARE_GPT_DATASET_PATH = os.environ.get(
    "SHARE_GPT_DATASET_PATH", os.path.expanduser("~/share_gpt.json")
)

common_model_paths = os.environ.get(
    "MODEL_PATHS",
    ["ibm-granite/granite-3.2-8b-instruct"],
)
common_batch_sizes = os.environ.get("BATCH_SIZES", [1, 2, 4, 8])
common_seq_lengths = os.environ.get("SEQ_LENGTHS", [64, 2048])
common_max_new_tokens = os.environ.get("MAX_NEW_TOKENS", [128])

output_dir = os.environ.get("OUTPUT_PATH", "/tmp/output")

# pass custom model path list for eg: EXPORT FMS_TESTING_COMMON_MODEL_PATHS="/tmp/models/granite-3-8b-base,/tmp/models/granite-7b-base"
if isinstance(common_model_paths, str):
    common_model_paths = common_model_paths.split(",")

# pass custom common batch sizes as a comma separated str of ints
if isinstance(common_batch_sizes, str):
    common_batch_sizes = [int(bs) for bs in common_batch_sizes.split(",")]

# pass custom common seq lengths as a comma separated str of ints
if isinstance(common_seq_lengths, str):
    common_seq_lengths = [int(sl) for sl in common_seq_lengths.split(",")]

# pass custom common max new tokens as a comma separated str of ints
if isinstance(common_max_new_tokens, str):
    common_max_new_tokens = [int(mnt) for mnt in common_max_new_tokens.split(",")]

common_shapes = list(
    itertools.product(
        common_model_paths,
        common_batch_sizes,
        common_seq_lengths,
        common_max_new_tokens,
    )
)

def __prepare_inputs(batch_size, seq_length, tokenizer, seed=0):
    prompts_and_sizes = sample_sharegpt_requests(
        SHARE_GPT_DATASET_PATH,
        batch_size,
        tokenizer,
        int(seq_length / 2),
        seq_length,
        seed,
    )
    ## TODO: for each prompt 
    prompt_list = []
    for prompt, _ in prompts_and_sizes:
        prompt_list.append(ids_for_prompt(prompt, tokenizer))

    input_ids, padding_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)
    return input_ids, padding_kwargs

def __infer_layer(model, max_len, device, max_new_tokens, batch_size, tokenizer):
    """
    Infer a model with registered layer hooks using generated inputs.

    Args:
        model (nn.Module): The model to infer.
        max_len (int): The maximum length of the input sequence.
        device (str): The device to use for inference.
        max_new_tokens (int): The maximum number of new tokens to generate.
        batch_size (int): The batch size for inference.
        tokenizer (Tokenizer): The tokenizer to use for encoding inputs.

    Returns:
        torch.Tensor: The inferred layer output.
    """
    
    do_sample = False
    use_cache = True

    prompts = __prepare_inputs(batch_size, max_len, tokenizer)
    ids, pad_input_ids = prompts

    if "cuda" in device:
        ids = ids.to("cuda")
    
    if hasattr(model.config, "ntk_scaling") and model.config.ntk_scaling:
        max_seq_len = max(max_len, model.config.max_expected_seq_len)
    else:
        # without ntk scaling, extending the seq length too far gives bogus results.
        max_seq_len = model.config.max_expected_seq_len

    if "generate" in mode:
        with torch.no_grad():
            result = generate(
                model,
                ids,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                do_sample=do_sample,
                max_seq_len=max_seq_len,
                timing="e2e",
                eos_token_id=None,
                contiguous_cache=True,
                extra_kwargs={},
            )
            result, timings = result
            logger.info(f"Generation completed: Result len is {len(result)}")
            if len(result.shape) == 1:
                result = result.unsqueeze(0)
    else:
        result = model.forward(
            ids,
            use_cache=use_cache
            )
        logger.info(f"Model forward completed: Result len is {len(result)}")

def __register_call_layers(model, batch_size, device, seq_length, max_new_tokens, tokenizer):
    """
    This function registers hooks on the model to track the forward pass of each layer.
    It returns a list of tuples containing the name and output of each layer in the model.

    Args:
        model (nn.Module): The PyTorch model to be analyzed.
        batch_size (int): The batch size used for inference.
        device (torch.device): The device on which the model is running.
        seq_length (int): The maximum sequence length of the input data.
        max_new_tokens (int): The maximum number of new tokens to be generated during inference.
        tokenizer (Tokenizer): The tokenizer used for tokenization.

    Returns:
        list: A list of tuples containing the name and output of each layer in the model.
    """
    layer_stack = []
    pt_compile_model_time = time.time()

    module_depth = {}
    module_name = {}

    def register_depths(module, current_depth=0, name='model'):
        module_depth[module] = current_depth
        module_name[module] = name
        parent=name
        # if we are dealing with array of layers
        array_layers = all(key.isdigit() for key in module._modules.keys())
        for name, child in module._modules.items():
            if array_layers: 
                register_depths(child, current_depth + 1, parent+'['+name+']')
            else:
                register_depths(child, current_depth + 1, parent+'.'+name)

    register_depths(model)

    def wrap_forward(layer):
        original_forward = layer.forward

        def safe_forward(*args, **kwargs):
            try:
                return original_forward(*args, **kwargs)
            except (RuntimeError,TypeError) as e:
                logger.error(f"Error in {layer.__class__.__name__}: {e}")
                return torch.zeros_like(args[0]) if args else None
        layer.forward = safe_forward
        

    hooks = []
    def pre_hook_fn(module, input):
        depth = module_depth.get(module, 0)
        layer_name = module_name.get(module, 0)
        prefix = '│    ' * depth
        if len(input) == 0: return
        input_shape_str = f"[{', '.join(map(str, input[0].shape))}]"
        input_type = str(input[0].dtype)
        if module.parameters() == None: return
        param_size = sum(p.numel() for p in module.parameters() if p.requires_grad)
        param_size_str = f"{param_size:,}" if param_size > 0 else "--"
        logger.info(f"{prefix}├─{layer_name}() -> {module.__class__.__name__} : | Input(arg): {input_shape_str} | {input_type} | Params: {param_size_str}")
        wrap_forward(module)
        # save input for later use with outputs
        module._debug_input = input 

    def post_hook_fn(module, input, output):
        layer_name = module_name.get(module, 0)
        # Save inputs and outputs
        if hasattr(module, '_debug_input'):
            layer_stack.append((layer_name, output))
            # Clean up
            delattr(module, '_debug_input')
    
    for name, layer in model.named_modules():
        hooks.append(layer.register_forward_pre_hook(pre_hook_fn))
        hooks.append(layer.register_forward_hook(post_hook_fn))

    
    __infer_layer(model= model, max_len=seq_length, 
                  device=device, max_new_tokens=max_new_tokens, 
                  batch_size=batch_size, tokenizer=tokenizer)

    for hook in hooks:
        hook.remove()

    pt_compile_model_time = time.time() - pt_compile_model_time
    logger.info(f"PT compile complete, took {pt_compile_model_time:.3f}s")

    return layer_stack

def get_metric_values(metric_list):
    if isinstance(metric_list, list):
        metric_shape = metric_list[0].shape
        metric_list_res = metric_list
    else:
        metric_shape = metric_list.shape
        metric_list_res = metric_list.flatten().tolist()

    return metric_list_res, metric_shape

def write_csv(values, path, metric, gpu_layer_shape, cpu_layer_shape, output_shape):
    """
    Write values to a CSV file at the given path.

    Args:
        values (list or float): A list of values to be written to the CSV file. 
        If `values` is a single float, it will be written as a scalar value in the first column of the CSV file.
        path (str): The path to the CSV file to write to.
        metric (str): The name of the metric being evaluated.
        gpu_layer_shape (tuple): The shape of the GPU layer used for training.
        cpu_layer_shape (tuple): The shape of the CPU layer used for training.
        output_shape (tuple): The shape of the output generated by the model.

    Returns:
        None
    """
    with open(path, 'w') as f:
        f.write(f'{metric}\n')
        f.write(f'GPU shape {gpu_layer_shape} CPU shape {cpu_layer_shape}\n')
        f.write(f'Metric shape {output_shape}\n')
        if not isinstance(values, float):
            for t in values:
                f.write(f"{t}\n")
        else:
            f.write(f"{values}\n")
        f.close()

def generate_layers_metrics(model_path, batch_size, seq_length, max_new_tokens):
    """
    Generate metrics for layers in a given model.

    Args:
        model_path (str): The path to the Hugging Face model.
        batch_size (int): The batch size used for inference.
        seq_length (int): The sequence length used for inference.
        max_new_tokens (int): The maximum number of new tokens allowed for generation.

    Returns:
        None
    """
    
    torch.manual_seed(42)
    os.environ["COMPILATION_MODE"] = "offline_decoder"

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/tmp/models/hf_cache"

    model_path_kwargs = {"variant": model_path}
    micro_model_kwargs = {"architecture": "hf_pretrained"}

    get_model_kwargs = {
        **model_path_kwargs,
        **micro_model_kwargs,
    }

    tokenizer = tokenizers.get_tokenizer(model_path)

    # prepare the cpu model
    validation_model = get_model(
        device_type="cpu",
        data_type=torch.float32,
        fused_weights=False,
        **get_model_kwargs,
    )

    # prepare the cuda model
    validation_model_cuda = get_model(
        device_type="cuda",
        data_type=torch.float16,
        fused_weights=False,
        **get_model_kwargs,
    )

    layer_stack_cpu = __register_call_layers(model=validation_model,
                                            batch_size=batch_size, 
                                            device="cpu", 
                                            seq_length=seq_length, max_new_tokens=max_new_tokens, 
                                            tokenizer=tokenizer)
    
    layer_stack_cuda = __register_call_layers(model=validation_model_cuda,
                                             batch_size=batch_size, 
                                             device="cuda", 
                                             seq_length=seq_length, max_new_tokens=max_new_tokens, 
                                             tokenizer=tokenizer)

    assert len(layer_stack_cuda) == len(layer_stack_cpu)

    for layer, cuda_output in layer_stack_cuda:
        tensor_cpu_out = None
        tensor_cuda_out = None
        cos = nn.CosineSimilarity(dim=-1)
        for cpu_layer, cpu_output in layer_stack_cpu:
            if cpu_layer == layer:
                logger.info("CPU Layer {} GPU Layer {}".format(cpu_layer, layer))

                if type(cpu_output) is tuple and type(cuda_output) is tuple:
                    cos_sim = []
                    abs_diff = []
                    if len(cpu_layer) < 2 and len(cpu_layer[-1]) == 1:
                        tensor_cuda_out = cuda_output[-1]
                        tensor_cpu_out = cpu_layer[-1]
                        for i in range(len(cpu_layer)):
                            logger.debug(f"inputs: {cuda_output[i].shape} {cpu_output[i].to('cuda').shape}")
                            cos_sim.append(cos(cuda_output[i], cpu_output[i].to('cuda')))
                            logger.debug(f"cos_sim output:{cos(cuda_output[i], cpu_output[i].to('cuda')).shape}")
                            abs_diff.append(torch.abs(cuda_output[i] - cpu_output[i].to('cuda')))
                    else:
                        head_tensor_cpu = cpu_output[-1]
                        head_tensor_gpu = cuda_output[-1]
                        for i in range(len(head_tensor_gpu)):
                            if isinstance(head_tensor_gpu[i], tuple):
                                for j in range(len(head_tensor_gpu[i])):
                                    tensor_cuda_out = head_tensor_gpu[i][j]
                                    tensor_cpu_out = head_tensor_cpu[i][j]
                                    logger.debug(f"inputs: {head_tensor_gpu[i][j].shape} {head_tensor_cpu[i][j].to('cuda').shape}")
                                    cos_sim.append(cos(head_tensor_cpu[i][j].to('cuda'), head_tensor_gpu[i][j]))
                                    logger.debug(f"cos_sim output:{cos(head_tensor_cpu[i][j].to('cuda'), head_tensor_gpu[i][j]).shape}")
                                    abs_diff.append(torch.abs(head_tensor_cpu[i][j].to('cuda') - head_tensor_gpu[i][j]))
                            else:
                                tensor_cuda_out = head_tensor_gpu[i]
                                tensor_cpu_out = head_tensor_cpu[i]
                                logger.debug(f"inputs: {head_tensor_gpu[i].shape} {head_tensor_cpu[i].to('cuda').shape}")
                                cos_sim.append(cos(head_tensor_cpu[i].to('cuda'), head_tensor_gpu[i]))
                                logger.debug(f"cos_sim output:{cos(head_tensor_cpu[i].to('cuda'), head_tensor_gpu[i]).shape}")
                                abs_diff.append(torch.abs(head_tensor_cpu[i].to('cuda') - head_tensor_gpu[i]))
                else:
                    tensor_cpu_out = cpu_output.to('cuda')
                    tensor_cuda_out = cuda_output
                    abs_diff = torch.abs(tensor_cpu_out - cuda_output)
                    cos_sim = cos(tensor_cpu_out, cuda_output)

                prefix = get_default_validation_prefix(model_path, max_new_token, batch_size, seq_length, 'float16')
                layer_name = str(layer).replace('[','').replace(']', '')

                abs_diff_path = os.path.join(output_dir, f"{prefix}--{layer_name}.abs_diff.csv")
                cos_sim_path = os.path.join(output_dir, f"{prefix}--{layer_name}.cos_sim.csv")

                cos_sim_res, cos_shape = get_metric_values(cos_sim)
                abs_diff_res, abs_diff_shape = get_metric_values(abs_diff)

                if not os.path.exists(abs_diff_path):
                    logger.debug("saving abs_diff files")
                    write_csv(abs_diff_res, abs_diff_path, "abs_diff", tensor_cuda_out.shape, tensor_cpu_out.shape, abs_diff_shape)
                if not os.path.exists(cos_sim_path):
                    logger.debug("saving cos_sim files")
                    write_csv(cos_sim_res, cos_sim_path, "cos_sim", tensor_cuda_out.shape, tensor_cpu_out.shape, cos_shape)

    logger.info(f"Completed {model_path} layers' metrics generation")

for model_id, batch_size, sequence_length, max_new_token in common_shapes:
    logger.info(f"testing model_id-{model_id}, max_new_tokens-{max_new_token}, batch_size-{batch_size}, seq_length-{sequence_length}")
    generate_layers_metrics(model_path=model_id, batch_size=batch_size, seq_length=sequence_length, max_new_tokens=max_new_token)
