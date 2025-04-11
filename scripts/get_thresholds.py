import glob
import os
import numpy as np

models = ["ibm-granite--granite-20b-code-instruct-8k"]
metrics = ["diff_mean", "ce"]

file_base = "/gpfs/users/jmrosenk/fullsize_models"

for model in models:
    for metric in metrics:
         
        path = os.path.join(file_base, f"{model}*{metric}*.csv")
        metric_files = glob.glob(path)

        metric_list = []
        for metric_file in metric_files:

            with open(metric_file, "r") as file:
                next(file)
                for line in file:
                    metric_list.append(float(line))
        print(f"found {len(metric_files)} metric files")
        if metric == "diff_mean":
            m1 = np.percentile(metric_list, .5)
            m2 = np.percentile(metric_list, 99.5)
            print(model, metric, m1, m2)
        else:
            print(model, metric, np.percentile(metric_list, 99.0))
