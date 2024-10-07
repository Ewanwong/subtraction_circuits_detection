"""
Generate data for multi-op equations.
"""
import argparse
import jsonlines
import os
import random

from utils.equations import MultiOpEquation, Equation
from utils.logger import log
from utils.tokens import get_ip_range_from_tokenizer

from tqdm.auto import tqdm
from banks import Prompt


def save_jsonl(data, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    with jsonlines.open(os.path.join(save_dir, filename), 'w') as writer:
        writer.write_all(data)


def get_max_samples_and_ranges(tokenizer_path, n_ops, max_samples):
    in_vocab_ip_range = get_ip_range_from_tokenizer(tokenizer_path, oov=False)
    oov_ip_range = get_ip_range_from_tokenizer(tokenizer_path, oov=True)
    max_pos_samples = {
        'oov': min(max_samples, (oov_ip_range['end'] - oov_ip_range['start'] + 1) ** (n_ops+1)),
        'in_vocab': min(max_samples, (in_vocab_ip_range['end'] - in_vocab_ip_range['start'] + 1) ** (n_ops+1))
    }
    log.info(f"Max possible samples: {max_pos_samples}")
    return max_pos_samples, in_vocab_ip_range, oov_ip_range


def generate_data(max_samples, operator, ip_range, max_op_per_eq):
    log.info(f"Generating data for operator: {operator}")
    log.info(f"Input range: {ip_range}")
    log.info(f"Max samples: {max_samples}")
    log.info(f"Max operators per equation: {max_op_per_eq}")

    raw_eqs = []
    repeat_cnt = 0
    tq = tqdm(total=max_samples)
    tq.set_description("Generating data...")

    while len(raw_eqs) < max_samples:
        multi_op_eq = MultiOpEquation(num_operator=max_op_per_eq)
        eq = multi_op_eq.generate_equation(operator, ip_range, int_only=True if operator != '/' else False,
                                           ans_wrap='answer{num}')
        if eq not in raw_eqs:
            raw_eqs.append(eq)
            tq.update(1)
        else:
            repeat_cnt += 1

    # log.info(f"Total equations: {len(raw_eqs)}")
    log.info(f"Repeated equations: {repeat_cnt}")

    return raw_eqs


def generate_all_op_data(tokenizer_path, n_ops, max_samples):
    # all_data = {op: {"oov": [], "inv": []} for op in Equation.operators}
    data_oov = {op: [] for op in Equation.operators}
    data_inv = {op: [] for op in Equation.operators}
    max_pos_samples, inv_ip_range, oov_ip_range = get_max_samples_and_ranges(tokenizer_path, n_ops, max_samples)
    for curr_op, op_data in data_inv.items():
        #model,prompt,response,solution,size_shot,size_eq,prompt_operator_shot,prompt_operator_eq,oov,response_type,response_num
        data_inv[curr_op] = generate_data(max_pos_samples['in_vocab'], curr_op, inv_ip_range, n_ops)
        data_oov[curr_op] = generate_data(max_pos_samples['oov'], curr_op, oov_ip_range, n_ops)
    return data_inv, data_oov


def generate_prompt(query, ctx, p_template):
    query, answer = query.split("=")
    answer_num = answer.split("{")[-1].split("}")[0]
    prompt = p_template.text({"examples": ctx, "query": query.strip()})
    return prompt, query.strip(), format(float(answer_num), '.3f') if "." in answer_num else int(answer_num)


def _create_n_shot_data(q_data, ctx_data, p_template, nshot):
    is_same = q_data == ctx_data
    data = {"nshot_prompts": []}
    for idx, query in enumerate(q_data):
        if is_same:
            ctx_chunk = random.sample(ctx_data[:idx-1]+ctx_data[idx+1:], nshot)
        else:
            ctx_chunk = random.sample(ctx_data, nshot)
        prompt, query, answer = generate_prompt(query, ctx_chunk, p_template)
        data["nshot_prompts"].append({"prompt": prompt, "answer": answer})
    return data


def create_n_shot_data(data, p_template, nshot, q_context_op_map, max_q_size, save_dir=None, save_prefix=None):
    for q_op, q_op_data in data.items():
        if q_op in ["/", "div", "div_w"]:
            continue
        random.shuffle(q_op_data)
        for ctx_op in q_context_op_map[q_op]:
            q_data = q_op_data[:max_q_size]
            if q_op == ctx_op:
                ctx_data = q_op_data
            else:
                random.shuffle(data[ctx_op])
                ctx_data = data[ctx_op][:max_q_size]
            nshot_data = _create_n_shot_data(q_data, ctx_data, p_template, nshot)
            if save_dir:
                filename = f"{save_prefix}_q_{Equation.op_dict[q_op]}_ctx_{Equation.op_dict[ctx_op]}_{nshot}shot.jsonl"
                save_jsonl(nshot_data["nshot_prompts"], save_dir, filename)
            else:
                raise NotImplementedError("Returning just the data is not implemented yet.")


def get_query_context_pairs():
    return {op: Equation.operators for op in Equation.operators}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_operators", type=int, required=True, help="Number of operators in the equation")
    ap.add_argument("--n_shot", type=int, required=True, help="Number of example equations to generate")
    ap.add_argument("--max_op_samples", type=int, required=True, help="Size of the dataset")
    ap.add_argument("--output_dir", type=str, required=True, help="Dir Path to the dataset")
    ap.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer file")
    ap.add_argument("--prompt_template_path", type=str, required=True, help="Prompt template file path")
    args = ap.parse_args()

    prompt_template = open(args.prompt_template_path, "r").read()
    p = Prompt(prompt_template)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #model,prompt,response,solution,size_shot,size_eq,prompt_operator_shot,prompt_operator_eq,oov,response_type,response_num
    all_data_inv, all_data_oov = generate_all_op_data(args.tokenizer_path, args.num_operators, args.max_op_samples)
    # log.info all data lens per op in a for loop
    for op in all_data_inv:
        log.info(f"Operator: {op}, In-vocab data: {len(all_data_inv[op])}, OOV data: {len(all_data_oov[op])}")
    merged_all_data = {op: all_data_inv[op]+all_data_inv[op] for op in Equation.operators}
    query_context_ops = get_query_context_pairs()

    create_n_shot_data(all_data_oov, p, args.n_shot, query_context_ops, args.max_op_samples, args.output_dir,
                       f"oov_{args.num_operators}op")
    create_n_shot_data(all_data_inv, p, args.n_shot, query_context_ops, args.max_op_samples, args.output_dir,
                       f"inv_{args.num_operators}op")
