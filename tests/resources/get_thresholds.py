import glob
import numpy as np
import argparse
import os
import math

import logging

import json

logger = logging.getLogger(__name__)
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(message)s")

parser = argparse.ArgumentParser(
    description="Script to get thresholds metrics"
)

parser.add_argument(
    "--models",
    type=str,
    default=[],
    nargs='+',
    required=True,
    help="List of models id separated by space. Eg.: ibm-granite/granite-20b-code-instruct-8k /tmp/models/granite-20b-code-cobol-v1"
)
parser.add_argument(
    "--metrics",
    type=str,
    default=[],
    nargs='+',
    required=True,
    help="List of metrics separated by space. Eg.: diff_mean ce",
)
parser.add_argument(
    "--file_base",
    type=str,
    default="/tmp/aiu-fms-testing-utils/output",
    required=True,
    help="Path where the thresholds output from the generate_metrics.py script were stored.",
)
parser.add_argument(
    "--layer_io",
    action="store_true",
    help="Sets the metric generation mode to layers IO"
)
parser.add_argument(
    "--output_path",
    type=str,
    default="/tmp/aiu-fms-testing-utils/output",
    required=False,
    help="Path where the json thresholds output for the layers will be saved.",
)
args = parser.parse_args()
models = [model.replace("/", "--") for model in args.models]
metrics = [metric for metric in args.metrics]
file_base = args.file_base
layer_mode = args.file_base if args.file_base else False

def sort_list_of_dictionaries(data):
    """Sorts a list of dictionaries alphabetically by key.

    Args:
        data: A list of dictionaries, each with a single key-value pair.

    Returns:
        A new list of dictionaries sorted alphabetically by key.
    """
    # Extract keys and store them in a separate list
    keys = [list(d.keys())[0] for d in data]

    # Get the indices that would sort the keys
    sorted_indices = sorted(range(len(keys)), key=lambda k: keys[k])

    # Reconstruct the list in the sorted order
    sorted_data = [data[i] for i in sorted_indices]
    return sorted_data

for model in models:
    result_dict = {"model_id": model}
    for metric in metrics:
        path = os.path.join(file_base, f"{model}*{metric}*.csv")
        metric_files = glob.glob(path)
        result_dict[metric] = []
        metric_list = []

        if not layer_mode:
            for metric_file in metric_files:

                with open(metric_file, "r") as file:
                    try:
                        next(file)
                        for line in file:
                            metric_list.append(float(line))
                    except StopIteration:
                        logger.info("No more metric files")
                        pass
            logger.info(f"found {len(metric_files)} metric files")
            logger.info(model, metric, np.percentile(metric_list, 99.0))
        else:
            layers = []
            for metric_file in metric_files:
                layer_dict = {}
                layer_name = metric_file.split("--")[-1].replace(".{}".format(metric), "")
                layer_name = layer_name.replace(".csv","")
                metric_layer_list = []
                with open(metric_file, "r") as file:
                    try:
                        for _ in range(3):
                            next(file)
                        for line in file:
                            metric_layer_list.append(float(line))
                    except StopIteration:
                        logger.info("No more metric files")
                        pass
                layer_dict[layer_name] = metric_layer_list
                layers.append(layer_dict)
            logger.info(f"found {len(layers)} layers metric files")

            for l in layers:
                for key in l.keys():
                    tmp = {}
                    if "abs_diff" in metric:
                        metric_val = np.linalg.norm(l[key])
                    else:
                        metric_val = np.mean(l[key])
                    tmp[key] = metric_val if not math.isnan(metric_val) else 0.0
                    result_dict[metric].append(tmp)
                    logger.info(f"Layer {key} avg {metric} = {metric_val}")
            result_dict[metric] = sort_list_of_dictionaries(result_dict[metric])
    json_output_path = args.output_path if args.output_path else file_base
    f_result_path = os.path.join(json_output_path, f"{model}-thresholds.json")
    with open(f_result_path, 'w') as fp:
        json.dump(result_dict, fp)

