import os


import torch
from torch import Tensor
from rich import print as rprint
import numpy as np

import json
import re

# Define the path to the results directory
results_dir = "../results"

for op in ["add", "sub","mul","div"]:
    results_dict = {}
    error_list = []
    # Iterate over each folder in the results directory
    for folder_name in os.listdir(results_dir):
        if "qwen" in folder_name:
            continue
        results_dict[folder_name] = {}
        folder_path = os.path.join(results_dir, folder_name)
        # Check if the path is a directory
        if os.path.isdir(folder_path):
            # Iterate over each file in the folder
            for ctx in ["add", "sub","mul","div"]:
                file_name = f"inv_1op_q_{op}_ctx_{ctx}_5shot_results.jsonl"
                file_path = os.path.join(folder_path, file_name)
                if not "inv" in file_name:
                    continue
                # Check if the path is a file
                data_list = []
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as file:
                        for line in file:
                            j_line = json.loads(line)
                            #print(j_line.keys())
                            correct, predicted = float(j_line["gold_answer"]), j_line["generated_ans"]
                            try:
                                correct = float(correct)
                                predicted = float(predicted)
                                data_list.append([correct, predicted])
                            except:
                                clean_predicted = re.sub(r'[^\d-]', '', predicted)
                                try:
                                    clean_predicted = float(clean_predicted)
                                    data_list.append([correct, clean_predicted])
                                except ValueError:
                                    error_list.append(f"Model: {folder_name}, Output:{predicted}")
                                    pass
                                    #print(f"Skipping line due to conversion error: {predicted}")
                arr = np.array(data_list)
                diff = (arr[:,0] - arr[:,1])**2
                diff_off_abs = np.abs(arr[:,0]) - np.abs(arr[:,1])

                mse = np.mean(diff)
                std_mse = np.std(diff)
                acc = np.mean(arr[:,0] == arr[:,1])

                mse_abs = np.mean(diff_off_abs**2)
                mse_abs_std = np.std(diff_off_abs**2)
                acc_abs = np.mean(np.abs(arr[:,0]) == np.abs(arr[:,1]))

                results_dict[folder_name][ctx] = {   "diff": diff,
                    "diff_off_abs": diff_off_abs,
                    "mse": mse,
                    "mse_std": std_mse,
                    "acc": acc,
                    "mse_abs": mse_abs,
                    "mse_abs_std": mse_abs_std,
                    "acc_abs": acc_abs
                }

                rprint(f"Results for {folder_name} and Context {ctx}")
                rprint(f"Mean Squared Error: {mse}")
                rprint(f"Standard Deviation of MSE: {std_mse}")
                rprint(f"Accuracy: {acc}")
                rprint("")

                rprint(f"Mean Squared Error (Absolute): {mse_abs}")
                rprint(f"Standard Deviation of MSE (Absolute): {mse_abs_std}")
                rprint(f"Accuracy (Absolute): {acc_abs}")

                rprint("")

    np.save(f"plots/{op}_res_for_plots.npy", results_dict)
    error_list = "\n".join(error_list)
    with open(f"errors/{op}_error_list.txt", "w") as error_file:
        error_file.write(error_list)