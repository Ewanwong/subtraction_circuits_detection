from transformers import GPT2Tokenizer, AutoTokenizer

model_name_or_path = 'gpt2-xl'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
MAX_LENGTH = 16

path = "data/gpt2-medium-all/inv_1op_q_sub_zero_shot.jsonl"

import datasets
import jsonlines
from tqdm import tqdm

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


def tokenize_function(example):
    # Concatenate instruction and completion
    text = example['prompt'] + str(example['answer'])
    tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=MAX_LENGTH)
    
    # Create labels
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    
    # Find the index where the completion starts
    instruction_tokens = tokenizer(example['prompt'], truncation=True, padding='max_length', max_length=MAX_LENGTH)['input_ids']
    instruction_length = len(instruction_tokens)
    
    # Mask out instruction tokens in labels
    labels = [-100]*instruction_length + input_ids[instruction_length:]
    
    # Ensure labels are the same length as input_ids
    labels = labels[:MAX_LENGTH]
    if len(labels) < MAX_LENGTH:
        labels += [-100]*(MAX_LENGTH - len(labels))
    
    tokenized['labels'] = labels
    return tokenized

trn_dataset, val_dataset, tst_dataset = jsonlines_to_hf_dataset_splits(path)

from datasets import Dataset

# Create a Dataset object
train_dataset = Dataset.from_list(trn_dataset)
val_dataset = Dataset.from_list(val_dataset)
tst_dataset = Dataset.from_list(tst_dataset)

# Apply the tokenization
tokenized_dataset = train_dataset.map(tokenize_function, batched=False)
val_tokenized_dataset = val_dataset.map(tokenize_function, batched=False)
tst_tokenized_dataset = tst_dataset.map(tokenize_function, batched=False)

from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=f'./{model_name_or_path}-subtraction-no-sft',
    num_train_epochs=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=1e-5,
    save_strategy='steps',
    save_steps=2000,
    save_total_limit=2,
    evaluation_strategy='steps',
    eval_steps=500,
    logging_steps=500,
    do_eval=True,
    
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=val_tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()