import argparse

import datasets
import jsonlines

from tqdm import tqdm
from trl import SFTConfig, SFTTrainer

from utils.logger import log


def split_data(data: list):
    train_data, other_data = data[:int(len(data) * 0.75)], data[int(len(data) * 0.75):]
    val_data, test_data = other_data[:int(len(other_data) * 0.5)], other_data[int(len(other_data) * 0.5):]
    return train_data, val_data, test_data


def jsonlines_to_hf_dataset_splits(jsonlines_path: str):
    with jsonlines.open(jsonlines_path) as reader:
        data = [doc for doc in reader]
    train_data, val_data, test_data = split_data(data)
    trn_dataset = datasets.load_dataset("json", data=train_data)
    val_dataset = datasets.load_dataset("json", data=val_data)
    tst_dataset = datasets.load_dataset("json", data=test_data)
    return trn_dataset, val_dataset, tst_dataset



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="Dir Path to the dataset")
    args = ap.parse_args()

    with jsonlines.open(args.path) as reader:
        for idx, document in tqdm(enumerate(reader)):
            log.info(document)
            if idx > 10:
                break