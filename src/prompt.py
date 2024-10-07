import argparse
import os
import jsonlines

import sglang as sgl

from glob import glob
from more_itertools import chunked
from tqdm.auto import tqdm

from utils.inference import get_default_prompting_params, math_complete


def read_jsonl(file):
    data = []
    with jsonlines.open(file, "r") as reader:
        for obj in reader:
            data.append(obj)
    return data


def get_and_check_answer(result, prompt, answer):
    gen_answer = result.text()
    gen_answer = gen_answer.replace(prompt["prompt"], "").split("}")[0].strip()
    gen_answer = gen_answer.replace(",", "")
    try:
        gen_answer = float(gen_answer) if "." in str(answer) else int(float(gen_answer))
    except ValueError:
        pass
    return gen_answer, gen_answer == answer


def _compile_jsonl_results(results, prompts, gold_answers, meta_info):
    jsonl_results = []
    corr_cnt = 0
    for p, ans, res, num_comp_tokens, op_logp, ip_logp in zip(prompts, gold_answers, results,
                                                              meta_info["num_completion_tokens"],
                                                              meta_info["output_token_logprobs"],
                                                              meta_info["input_token_logprobs"]):
        gen_ans, is_correct = get_and_check_answer(res, p, ans)
        corr_cnt += 1 if is_correct else 0
        jsonl_results.append({"query_prompt": p["prompt"],
                              "gold_answer": ans,
                              "generated_ans": gen_ans,
                              "is_correct": is_correct,
                              "meta": {"num_completion_tokens": num_comp_tokens,
                                       "output_token_logprobs": op_logp,
                                       "input_token_logprobs": ip_logp}})
    return jsonl_results, corr_cnt


def infer_and_save_to_jsonl(input_data, out_file):
    prompts = [{"prompt": data["prompt"]} for data in input_data]
    gold_answers = [data["answer"] for data in input_data]
    chunked_prompts = list(chunked(prompts, args.batch_size))
    results = []
    meta_info = {"num_completion_tokens": [], "output_token_logprobs": [], "input_token_logprobs": []}
    for cp in tqdm(chunked_prompts):
        res = math_complete.run_batch(cp)
        results.extend(res)
        for r in res:
            result_meta = r.get_meta_info("answer")
            meta_info["num_completion_tokens"].append(result_meta["completion_tokens"])
            meta_info["output_token_logprobs"].append(result_meta["output_token_logprobs"])
            meta_info["input_token_logprobs"].append(result_meta["input_token_logprobs"])

    jsonl_res, corr_cnt = _compile_jsonl_results(results, prompts, gold_answers, meta_info)
    with jsonlines.open(out_file, "w") as writer:
        writer.write_all(jsonl_res)
    return corr_cnt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Dir Path to the dataset")
    ap.add_argument("--filter_prefix", type=str, required=True, help="Prefix to filter the files")
    ap.add_argument("--model", type=str, default="llama-3.1-70b", required=True, help="Model to use for inference")
    ap.add_argument("--output_dir", type=str, required=True, help="Dir Path to the output")
    ap.add_argument("--batch_size", type=int, default=32, required=False, help="Batch size for inference")
    ap.add_argument("--port", type=int, default=12323, required=False, help="Port to connect to the server")
    ap.add_argument("--api_key", type=str, default="sk_noreq", required=False, help="API key to connect")
    args = ap.parse_args()

    total_files = glob(f"{args.data_dir}*/{args.filter_prefix if args.filter_prefix else ''}*.jsonl", recursive=True)
    total_files = [file for file in total_files if args.model in file]
    print(total_files)

    prompting_params = get_default_prompting_params()
    sgl.set_default_backend(sgl.RuntimeEndpoint(f"http://localhost:{args.port}", api_key=f"{args.api_key}"))
    os.makedirs(os.path.join(args.output_dir, args.model), exist_ok=True)

    for file in tqdm(total_files):
        print(f"Processing file: {file}")
        input_data = read_jsonl(file)
        out_file = os.path.join(args.output_dir, args.model, f"{os.path.basename(file).split('.')[0]}_results.jsonl")
        correct_cnt = infer_and_save_to_jsonl(input_data, out_file)
        accuracy = correct_cnt / len(input_data)
        print(f"Accuracy: {accuracy}")
        print(f"Results saved to: {out_file}")

