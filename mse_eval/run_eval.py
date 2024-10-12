import os


import torch
from torch import Tensor
from tqdm.notebook import tqdm
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from rich import print as rprint
from transformer_lens import utils, HookedTransformer, ActivationCache
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import partial
import re

torch.set_grad_enabled(False)

from ioi_dataset import NAMES, IOIDataset
from path_patching import Node, IterNode, path_patch, act_patch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
from datasets import load_dataset


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.4b-deduped")
data_path = "./data/llama-3.1-8b-all/"
data_files = ["inv_1op_q_add_ctx_add_5shot.jsonl", "inv_1op_q_sub_ctx_sub_5shot.jsonl"]

file_path = os.path.join(data_path, data_files[0])
dataset = load_dataset("json", data_files=file_path, split=None)

print(dataset)
for i in dataset:
    print(i)

data_list = []

for file_name in data_files:
    file_path = os.path.join(data_path, file_name)
    with open(file_path, 'r') as file:
        data_list.extend(json.loads(line) for line in file)

#tokenizer = AutoTokenizer.from_pretrained(model)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    #torch_dtype=torch.bfloat16,
    device="cuda:0"
)
prompt = [data_list[1]["prompt"]]
sequences = pipe(
    prompt,
    max_new_tokens=4,
    return_full_text = True
)
#print(data_list[0])
#print(sequences)
for seq in sequences:
    answer = seq[0]['generated_text'].split('answer{')[-1]
    #number_match = re.search(r'\d+', answer)
    number_match = re.search(r'-?\d+', answer)
    answer = int(number_match.group())
    print(answer)
    print(seq[0]['generated_text'].split('\n')[-1])
    #print(f"Result: {seq['generated_text']}")


