import argparse
import os

import datasets
import jsonlines
import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from utils.logger import log

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"] = "subtraction"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "false"

# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"


def split_data(data: list):
    train_data, other_data = data[:int(len(data) * 0.75)], data[int(len(data) * 0.75):]
    val_data, test_data = other_data[:int(len(other_data) * 0.5)], other_data[int(len(other_data) * 0.5):]
    return train_data, val_data, test_data


def jsonlines_to_hf_dataset_splits(jsonlines_path: str):
    with jsonlines.open(jsonlines_path) as reader:
        data = [doc for doc in reader]
    train_data, val_data, test_data = split_data(data)
    trn_dataset = datasets.Dataset.from_list(train_data)
    val_dataset = datasets.Dataset.from_list(val_data)
    tst_dataset = datasets.Dataset.from_list(test_data)
    return trn_dataset, val_dataset, tst_dataset


def formatting_prompts_func(example):
    text = f"{example['prompt']}{example['answer']}"
    return [text]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="Dir Path to the dataset")
    args = ap.parse_args()

    model_name = "gpt2-xl"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    trn_dataset, val_dataset, tst_dataset = jsonlines_to_hf_dataset_splits(args.path)

    response_template = "."
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model,
        train_dataset=trn_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(output_dir="finetuned_models/", report_to="wandb", num_train_epochs=600,
                       per_device_train_batch_size=1, per_device_eval_batch_size=8, save_strategy="steps",
                       save_steps=10000, eval_strategy="steps", eval_steps=2000, logging_steps=500, max_seq_length=64),
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer.train()
    wandb.finish()
