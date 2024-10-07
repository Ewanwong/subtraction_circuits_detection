import argparse
import os
import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from glob import glob

from utils.equations import Equation


model_to_name = {"llama-3.1-70b": "LLaMa-3-70B", "llama-3.1-8b": "LLaMa-3-8B",
                 "qwen2-math-7b": "Qwen2-Math-7B", "gpt2-medium": "GPT2-Medium",
                 "qwen2-math-72b": "Qwen2-Math-72B"}


"""def get_accuracy_and_total_jsonl(file):
    corr_cnt, total = 0, 0
    with jsonlines.open(file, "r") as reader:
        for obj in reader:
            corr_cnt += 1 if obj["is_correct"] else 0
            total += 1
    return corr_cnt/total, total


def get_neg_accuracy_and_total_jsonl(file):
    corr_cnt, total = 0, 0
    with jsonlines.open(file, "r") as reader:
        for obj in reader:
            if obj["is_correct"]:
                corr_cnt += 1
            elif type(obj["generated_ans"]) is int and obj["generated_ans"] == -obj["gold_answer"]:
                corr_cnt += 1
            total += 1
    return corr_cnt/total, total



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Path to the data directory")
    ap.add_argument("--model", type=str, required=True, help="Model name")
    ap.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_files = glob(f"{args.input_dir}/{args.model}/*results.jsonl", recursive=True)
    q_ops = [op for op in Equation.operators if op != "/"]
    ctx_ops = [op for op in Equation.operators]
    inv_res = np.zeros(shape=(len(q_ops), len(ctx_ops)))
    oov_res = np.zeros(shape=(len(q_ops), len(ctx_ops)))
    inv_neg_res = np.zeros(shape=(len(q_ops), len(ctx_ops)))
    oov_neg_res = np.zeros(shape=(len(q_ops), len(ctx_ops)))
    inv_no_samples, oov_no_samples = 0, 0
    inv_neg_samples, oov_neg_samples = 0, 0
    for res_file in total_files:
        neg_accuracy, neg_total = get_neg_accuracy_and_total_jsonl(res_file)
        accuracy, total = get_accuracy_and_total_jsonl(res_file)
        curr_q_op = res_file.split("_q_")[1].split("_")[0]
        q_op = Equation.op_reverse_dict[curr_q_op.strip()]
        curr_ctx_op = res_file.split("_ctx_")[1].split("_")[0]
        ctx_op = Equation.op_reverse_dict[curr_ctx_op.strip()]
        if "inv" in res_file:
            if not inv_no_samples:
                inv_no_samples = total
            else:
                assert inv_no_samples == total
            inv_res[q_ops.index(q_op)][ctx_ops.index(ctx_op)] = accuracy
            inv_neg_res[q_ops.index(q_op)][ctx_ops.index(ctx_op)] = neg_accuracy
        if "oov" in res_file:
            if not oov_no_samples:
                oov_no_samples = total
            else:
                assert oov_no_samples == total
            oov_res[q_ops.index(q_op)][ctx_ops.index(ctx_op)] = accuracy
            oov_neg_res[q_ops.index(q_op)][ctx_ops.index(ctx_op)] = neg_accuracy

    fig, ax = plt.subplots(2, 2)
    fig.suptitle(f'{model_to_name[args.model]} in-vocab samples: {inv_no_samples}, out-vocab samples: {oov_no_samples}', fontsize=10)
    sns.heatmap(inv_res.T, annot=True, ax=ax[0][0], xticklabels=q_ops, yticklabels=ctx_ops)
    ax[0][0].set_title(f"In-vocab accuracy", fontsize=8)
    sns.heatmap(inv_neg_res.T, annot=True, ax=ax[0][1], xticklabels=q_ops, yticklabels=ctx_ops)
    ax[0][1].set_title(f"In-vocab accuracy \n(ignoring negative sign)", fontsize=8)
    sns.heatmap(oov_res.T, annot=True, ax=ax[1][0], xticklabels=q_ops, yticklabels=ctx_ops)
    ax[1][0].set_xlabel(f"OOV accuracy", fontsize=8)
    sns.heatmap(oov_neg_res.T, annot=True, ax=ax[1][1], xticklabels=q_ops, yticklabels=ctx_ops)
    ax[1][1].set_xlabel(f"OOV accuracy \n(ignoring negative sign)", fontsize=8)

    plt.savefig(f"{args.output_dir}/{args.model}_results.pdf")"""


def get_neg_accuracy_and_total_jsonl(file):
    abs_corr_cnt, neg_ignored_corr_cnt, incorrect_cnt, total = 0, 0, 0, 0
    with jsonlines.open(file, "r") as reader:
        for obj in reader:
            if obj["gold_answer"] < 0:
                total += 1
                if obj["is_correct"]:
                    abs_corr_cnt += 1
                elif type(obj["generated_ans"]) is int and obj["generated_ans"] == -obj["gold_answer"]:
                    neg_ignored_corr_cnt += 1
                else:
                    incorrect_cnt += 1
    return abs_corr_cnt, neg_ignored_corr_cnt, incorrect_cnt, total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Path to the data directory")
    ap.add_argument("--model", type=str, required=True, help="Model name")
    ap.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_files = glob(f"{args.input_dir}/{args.model}/*results.jsonl", recursive=True)

    inv_corr_count, inv_total_samples = 0, 0
    oov_corr_count, oov_total_samples = 0, 0

    # FIXME: This is not generalized code, and it doesn't generate the plot yet.
    for res_file in total_files:
        curr_q_op = res_file.split("_q_")[1].split("_")[0]
        curr_ctx_op = res_file.split("_ctx_")[1].split("_")[0]

        if curr_q_op == "sub":
            abs_corr_cnt, neg_ignored_corr_cnt, fully_incorrect_cnt, total = get_neg_accuracy_and_total_jsonl(res_file)
            #print(f"File: {res_file}, Negative Accuracy: {neg_accuracy}, Total: {neg_total}")
            print(f"File: {res_file}, Correct (w Neg sign): {abs_corr_cnt}, Correct (Neg sign Ignored): {neg_ignored_corr_cnt}, "
                  f"Incorrect: {fully_incorrect_cnt}")

            # if "inv" in res_file:
            #     inv_abs_corr += neg_accuracy * neg_total
            #     inv_total_samples += neg_total
            # elif "oov" in res_file:
            #     oov_corr_count += neg_accuracy * neg_total
            #     oov_total_samples += neg_total

    # fig, ax = plt.subplots(figsize=(8, 6))
    # categories = ['NehIn-vocab', 'Out-of-vocab']
    # accuracies = [inv_final_accuracy, oov_final_accuracy]
    #
    # sns.barplot(x=categories, y=accuracies, palette="Blues", ax=ax)
    #
    # ax.set_ylim(0, 1)
    # ax.set_ylabel('Negative Accuracy')
    # ax.set_xlabel('Data Type')
    # ax.set_title(f'"sub" Operation ({model_to_name[args.model]} In-vocab: {inv_total_samples}, Out-of-vocab: {oov_total_samples})\nAccuracy Ratio (inv/oov): {accuracy_ratio:.2f}')
    #
    # for i, v in enumerate(accuracies):
    #     ax.text(i, v + 0.02, f"{v:.2f}", color='black', ha='center', fontsize=12)
    #
    # plt.tight_layout()
    # plt.savefig(f"{args.output_dir}/{args.model}_sub_results_comparison.pdf")
    #plt.show()
