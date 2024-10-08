from typing import Union, List, Optional
import warnings
import torch as t
import numpy as np
from transformers import AutoTokenizer
import random
import copy
import re
import json

def gen_flipped_prompts(prompts, flip_mode):
    flipped_prompts = []
    for prompt in prompts:
        if prompt['few_shot'] == False:
            correct = prompt["correct"]
            incorrect = prompt["incorrect"]
            few_shot = prompt["few_shot"]
            n_shot = prompt["n_shot"]
            n1 = prompt["n2"]
            n2 = prompt["n1"]
            question = f"### {n1} - {n2} = answer{{"
            flipped_prompts.append({"text": question,
                                      "few_shot": prompt['few_shot'],
                                      "n_shot": prompt['n_shot'],
                                      "query_examples_inputs": None,
                                      "query_examples_answers": None,
                                      "correct": correct, 
                                      "incorrect": incorrect, 
                                      "n1": n1, 
                                      "n2": n2})
        if flip_mode == "same":
            query_examples_inputs = prompt["query_examples_inputs"]
            query_examples_answers = prompt["query_examples_answers"]
        elif flip_mode == "different":
            query_examples_inputs = [[num[1], num[0]] for num in prompt["query_examples_inputs"]]
            query_examples_answers = [-num for num in prompt["query_examples_answers"]]
        else:
            raise ValueError("Flip mode must be either 'same' or 'different'")
        correct = prompt["correct"]
        incorrect = prompt["incorrect"]
        few_shot = prompt["few_shot"]
        n_shot = prompt["n_shot"]
        n1 = prompt["n2"]
        n2 = prompt["n1"]
        examples = [f"### {input[0]} - {input[1]} = answer{{{answer}}} ###<eop>\n" for input, answer in zip(query_examples_inputs, query_examples_answers)]
        question = f"### {n1} - {n2} = answer{{"
        text = "".join(examples) + question
        flipped_prompts.append({"text": text, 
                               "few_shot": few_shot,
                               "n_shot": n_shot,
                               "query_examples_inputs": query_examples_inputs,
                               "query_examples_answers": query_examples_answers,
                               "correct": correct, 
                               "incorrect": incorrect, 
                               "n1": n1, 
                               "n2": n2})
    return flipped_prompts


def get_number_idxs(prompts, tokenizer, idx_types=["n1", "n2"], prepend_bos=False):
    number_idx_dict = dict((idx_type, []) for idx_type in idx_types)
    for prompt in prompts:
        text_split = prompt["text"].split(" ")
        toks = tokenizer.tokenize(" ".join(text_split[:-1]))
        # Get the kast instance of the first number in the equation
        number_idx_dict["n1"].append(
            len(toks) - toks[::-1].index(tokenizer.tokenize(" " + prompt["n1"])[0]) - 1
        )
        # Get the last instance of second number in the equation
        number_idx_dict["n2"].append(
            len(toks) - toks[::-1].index(tokenizer.tokenize(" " + prompt["n2"])[0]) - 1
        )
       

    return [
        int(prepend_bos) + t.tensor(number_idx_dict[idx_type])
        for idx_type in idx_types
    ]

def get_word_idxs(prompts, word_list, tokenizer):
    """Get the index of the words in word_list in the prompts. Exactly one of the word_list word has to be present in each prompt"""
    idxs = []
    tokenized_words = [
        tokenizer.decode(tokenizer(word)["input_ids"][0]) for word in word_list
    ]
    for prompt in prompts:
        toks = [
            tokenizer.decode(t)
            for t in tokenizer(prompt["text"], return_tensors="pt", padding=True)[
                "input_ids"
            ][0]
        ]
        idx = None
        for i, w_tok in enumerate(tokenized_words):
            if word_list[i] in prompt["text"]:
                try:
                    idx = toks.index(w_tok)
                    if toks.count(w_tok) > 1:
                        # get the last time the work appears
                        idx = len(toks) - toks[::-1].index(w_tok) - 1
                except:
                    idx = toks.index(w_tok)
                    # raise ValueError(toks, w_tok, prompt["text"])
        if idx is None:
            raise ValueError(f"Word {word_list} and {i} not found {prompt}")
        idxs.append(idx)
    return t.tensor(idxs)


def get_end_idxs(toks, tokenizer, name_tok_len=1, prepend_bos=False):
    relevant_idx = int(prepend_bos)
    # if the sentence begins with an end token
    # AND the model pads at the end with the same end token,
    # then we need make special arrangements

    pad_token_id = tokenizer.pad_token_id

    end_idxs_raw = []
    for i in range(toks.shape[0]):
        if pad_token_id not in toks[i][1:]:
            end_idxs_raw.append(toks.shape[1])
            continue
        nonzers = (toks[i] == pad_token_id).nonzero()[relevant_idx][0].item()
        end_idxs_raw.append(nonzers)
    end_idxs = t.tensor(end_idxs_raw)
    end_idxs = end_idxs - 1 - name_tok_len

    for i in range(toks.shape[0]):
        assert toks[i][end_idxs[i] + 1] != 0 and (
            toks.shape[1] == end_idxs[i] + 2 or toks[i][end_idxs[i] + 2] == pad_token_id
        ), (
            toks[i],
            end_idxs[i],
            toks[i].shape,
            "the END idxs aren't properly formatted",
        )

    return end_idxs





def get_idx_dict(prompts, tokenizer, prepend_bos=False, toks=None):
    (n1_idxs, n2_idxs,) = get_number_idxs(
        prompts,
        tokenizer,
        idx_types=["n1", "n2"],
        prepend_bos=prepend_bos,
    )

    end_idxs = get_end_idxs(
        toks,
        tokenizer,
        name_tok_len=1,
        prepend_bos=prepend_bos,
    )

    #punct_idxs = get_word_idxs(prompts, [",", "."], tokenizer)
    subtraction_idxs = get_word_idxs(prompts, [" -"], tokenizer)

    return {
        "n1": n1_idxs,
        "n2": n2_idxs,
        "end": end_idxs,
        "starts": t.zeros_like(end_idxs),
        #"punct": punct_idxs,
        "subtraction": subtraction_idxs,
        "n1-1": n1_idxs - 1,
        "n1+1": n1_idxs + 1,
        "n2-1": n2_idxs - 1,
        "n2+1": n2_idxs + 1,
    }

def parse_prompts(prompts):
    # parse "### 255 - 758 = answer{-503} ### <eop>\n" into {"text": "255 - 758 =", "correct": "-", "incorrect": "503", "n1": 255, "n2": 758}
    """
    prompts = []
    parsed_prompts = []
    with open(jsonl_file, "r") as f:
        for line in f:
            prompts.append(json.loads(line))
    """
    parsed_prompts = []
    for prompt in prompts:
        assert prompt['gold_answer'] < 0, "Gold answer is not negative"
        few_shot = prompt['prompt'].count("<eop>\n") > 0 
        if few_shot:
            n_shot = prompt['prompt'].count("<eop>\n") - 1
            query_question = prompt["prompt"].split("<eop>\n")[-1]
            query_examples = prompt["prompt"].split("<eop>\n")[:-1]
            query_examples_numbers = [[int(num) for num in re.findall(r'-?\d+', example)] for example in query_examples]
            query_examples_inputs = [(num[0], num[1]) for num in query_examples_numbers]
            query_examples_answers = [num[2] for num in query_examples_numbers]
        else:
            n_shot = 0
            query_question = prompt["prompt"]
            query_examples_inputs = None
            query_examples_answers = None
            
        query_question_numbers = [int(num) for num in re.findall(r'-?\d+', query_question)]
        query_question_n1 = query_question_numbers[0]
        query_question_n2 = query_question_numbers[1]

        parsed_prompts.append({"text": prompt["prompt"], 
                               "few_shot": few_shot,
                               "n_shot": n_shot,
                               "query_examples_inputs": query_examples_inputs,
                               "query_examples_answers": query_examples_answers,
                               "correct": "-", 
                               "incorrect": str(-prompt['gold_answer']), 
                               "n1": str(query_question_n1), 
                               "n2": str(query_question_n2)})
    return parsed_prompts


class SubtractionDataset:
    def __init__(
        self,
        tokenizer=None,
        prompts=None,
        prepend_bos=False,
        device="cuda",
        flip_mode="same", # "same" or "different", same indicates the few shot examples are not corrupted, different indicates the order of numbers in each equation is swapped
    ):

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        if prompts is None:
            raise ValueError("Prompts must be provided by a jsonl file")
            self.prompts = gen_prompt_uniform(N=N, max_number=max_number) 
        elif isinstance(prompts, list) and isinstance(prompts[0], dict) and "gold_answer" in prompts[0].keys():
            self.prompts = parse_prompts(prompts)
        else:
            self.prompts = prompts

        self.sentences = [
            prompt["text"] for prompt in self.prompts
        ]

        texts = [
            (self.tokenizer.bos_token if prepend_bos else "") + prompt["text"]
            for prompt in self.prompts
        ]
        self.toks = t.Tensor(self.tokenizer(texts, padding=True).input_ids).long()

        self.word_idx = get_idx_dict(
            self.prompts,
            self.tokenizer,
            prepend_bos=prepend_bos,
            toks=self.toks,
        )
        self.prepend_bos = prepend_bos


        self.N = len(prompts)
        self.max_len = max([
            len(self.tokenizer(prompt["text"]).input_ids)
            for prompt in self.prompts
        ])

        self.correct_tokenIDs = [self.tokenizer.encode(" " + prompt["correct"])[0] for prompt in self.prompts]
        self.incorrect_tokenIDs = [self.tokenizer.encode(" " + prompt["incorrect"])[0] for prompt in self.prompts]

        self.tokenized_prompts = []

        for i in range(self.N):
            self.tokenized_prompts.append(
                "|".join([self.tokenizer.decode(tok) for tok in self.toks[i]])
            )

        self.device = device
        self.flip_mode = flip_mode
        self.to(device)
    

    def gen_flipped_prompts(self):
        
        # Get flipped prompts
        flipped_prompts = gen_flipped_prompts(self.prompts, self.flip_mode)

        flipped_dataset = SubtractionDataset(
                    tokenizer=self.tokenizer,
                    prompts=flipped_prompts,
                    prepend_bos=self.prepend_bos,
                    device=self.device,
                    flip_mode=self.flip_mode,
        )
        return flipped_dataset

    def copy(self):
        copy_ioi_dataset = SubtractionDataset(
                    tokenizer=self.tokenizer,
                    prompts=self.prompts.copy(),
                    prepend_bos=self.prepend_bos,
                    device=self.device,
                    flip_mode=self.flip_mode)
        return copy_ioi_dataset

    def __getitem__(self, key):
        sliced_prompts = self.prompts[key]
        sliced_dataset = SubtractionDataset(
                    tokenizer=self.tokenizer,
                    prompts=sliced_prompts,
                    prepend_bos=self.prepend_bos,
                    device=self.device,
                    flip_mode=self.flip_mode
        )
        return sliced_dataset

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        raise NotImplementedError()

    def __len__(self):
        return self.N

    def tokenized_prompts(self):
        return self.toks

    def to(self, device):
        self.toks = self.toks.to(device)
        return self



