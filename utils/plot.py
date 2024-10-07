import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


base_dir = r"C:\Users\ASUS\Desktop\NLP_HiWi\understand_llm_math\exploration\std_op\results"
output_base_dir = r"C:\Users\ASUS\Desktop\NLP_HiWi\understand_llm_math\exploration\std_op\figures"


def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


def calculate_logprobs(data):
    correct_logprobs = []
    incorrect_logprobs = []
    
    for sample in data:
        # Where first number is smaller than the second number
        if sample['gold_answer'] >= 0:
            continue

        logprobs = sample['meta']['output_token_logprobs']
        cumulative_logprob = 0
        for logprob, token_id, token in logprobs:
            if token_id == 92 and token == "}":
                break
            cumulative_logprob += logprob
        
        if sample['is_correct']:
            correct_logprobs.append(cumulative_logprob)
        else:
            incorrect_logprobs.append(cumulative_logprob)
    
    mean_correct = np.mean(correct_logprobs) if correct_logprobs else 0
    mean_incorrect = np.mean(incorrect_logprobs) if incorrect_logprobs else 0

    std_correct = np.std(correct_logprobs) if correct_logprobs else 0
    std_incorrect = np.std(incorrect_logprobs) if incorrect_logprobs else 0
    
    return (mean_correct, mean_incorrect), (std_correct, std_incorrect)


def plot_bar(means, std_devs, ax, title):
    categories = ['Correct', 'Incorrect']
    sns.barplot(x=categories, y=means, palette="Blues", ax=ax, ci=None)

    ax.errorbar(x=[0, 1], y=means, yerr=std_devs, fmt='none', c='black', capsize=5)
    
    ax.set_title(title)
    ax.set_ylabel('Mean Cumulative Logprob')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(means):
        ax.text(i, v, f"{v:.4f}", color='black', ha='center', va='bottom' if v > 0 else 'top')


def should_process_file(file_name, only_sub):
    if only_sub:
        return "_q_sub_" in file_name and "_ctx_sub_" in file_name
    return True 


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_sub", action='store_true', help="Process only 'sub' query and context files")
    args = ap.parse_args()

    for model_dir in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_dir)

        if os.path.isdir(model_path):
            inv_files = [f for f in os.listdir(model_path) if f.startswith("inv") and f.endswith(".jsonl")]
            oov_files = [f for f in os.listdir(model_path) if f.startswith("oov") and f.endswith(".jsonl")]

            for inv_file, oov_file in zip(inv_files, oov_files):
                if should_process_file(inv_file, args.only_sub):
                    inv_data = load_jsonl(os.path.join(model_path, inv_file))
                    oov_data = load_jsonl(os.path.join(model_path, oov_file))
                    (inv_means, inv_stds) = calculate_logprobs(inv_data)
                    (oov_means, oov_stds) = calculate_logprobs(oov_data)
                    query_type = inv_file.split('_')[3]
                    context_type = inv_file.split('_')[5]

                    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                    plot_bar(inv_means, inv_stds, axs[0], 'In-vocab')
                    plot_bar(oov_means, oov_stds, axs[1], 'Out-of-vocab')

                    fig.suptitle(f'Model: {model_dir}, Query: {query_type}, Context: {context_type}', fontsize=14)
                    plt.tight_layout()

                    output_dir = os.path.join(output_base_dir, model_dir)
                    os.makedirs(output_dir, exist_ok=True)
                    output_file_path = os.path.join(output_dir, inv_file.replace(".jsonl", "_vs_oov_sub.pdf"))
                    plt.savefig(output_file_path)
                    plt.close()