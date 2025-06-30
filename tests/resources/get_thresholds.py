import glob
import numpy as np
import argparse
import os

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
args = parser.parse_args()
models = [model.replace("/", "--") for model in args.models]
metrics = [metric for metric in args.metrics]
file_base = args.file_base
layer_mode = args.file_base if args.file_base else False

for model in models:
    for metric in metrics:
        path = os.path.join(file_base, f"{model}*{metric}*.csv")
        metric_files = glob.glob(path)

        if not layer_mode:
            metric_list = []
            for metric_file in metric_files:

                with open(metric_file, "r") as file:
                    next(file)
                    for line in file:
                        metric_list.append(float(line))
            print(f"found {len(metric_files)} metric files")
            print(model, metric, np.percentile(metric_list, 99.0))
        else:
            metric_list = []
            layers = []
            for metric_file in metric_files:
                layer_dict = {}
                layer_name = metric_file.split("--")[-1].replace(".{}".format(metric), "")
                with open(metric_file, "r") as file:
                    next(file)
                    for line in file:
                        metric_list.append(float(line))
                layer_dict[layer_name] = metric_list
                layers.append(layer_dict)
            print(f"found {len(layers)} layers metric files")

            for l in layers:
                for key in l.keys():
                    print(f"Layer {key} avg {metric}")
                    print(metric, np.percentile(l[key], 99.0))

