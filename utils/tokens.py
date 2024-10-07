from more_itertools import consecutive_groups

from tokenizers import Tokenizer


def get_all_num_tokens_from_tokenizer(tokenizer_path: str):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab = tokenizer.get_vocab()

    # https://stackoverflow.com/questions/44891070/whats-the-difference-between-str-isdigit-isnumeric-and-isdecimal-in-pyth
    num_tokens = [token for token in vocab.keys() if token.isdecimal()]
    return num_tokens


def get_ip_range_from_tokenizer(tokenizer_path: str, oov: bool, oov_range: int = 100):
    num_tokens = get_all_num_tokens_from_tokenizer(tokenizer_path)
    # print(f"Number of numeric tokens: {len(num_tokens)}")
    # print(sorted(num_tokens))
    nums = sorted([int(float(token)) for token in num_tokens])
    # print(f"No of Int tokens: {len(nums)}")
    num_set = list(set(nums))
    # print(f"No of Unique Int tokens: {len(num_set)}")
    consecutive = [list(group) for group in consecutive_groups(num_set)]
    # print(f"Consecutive groups: {consecutive}")
    start, end = (consecutive[0][0], consecutive[0][-1]) if not oov else (int(consecutive[0][-1]+1),
                                                                          int(consecutive[0][-1])*oov_range)
    return {'start': start, 'end': end}


if __name__ == "__main__":
    tokenizer_path = "tokenizer_data/gpt2-medium-tokenizer.json"
    oov = False
    ip_range = get_ip_range_from_tokenizer(tokenizer_path, oov)
    print(f"Input range in-vocab: {ip_range}")

    oov = True
    ip_range = get_ip_range_from_tokenizer(tokenizer_path, oov)
    print(f"Input range oov: {ip_range}")

    tokenizer_path = "./tokenizer_data/llama-2-7b-tokenizer.json"
    oov = False
    ip_range = get_ip_range_from_tokenizer(tokenizer_path, oov)
    print(f"Input range in-vocab: {ip_range}")

    oov = True
    ip_range = get_ip_range_from_tokenizer(tokenizer_path, oov)
    print(f"Input range oov: {ip_range}")

    tokenizer_path = "./tokenizer_data/llama-3.1-8b-tokenizer.json"
    oov = False
    ip_range = get_ip_range_from_tokenizer(tokenizer_path, oov)
    print(f"Input range in-vocab: {ip_range}")

    oov = True
    ip_range = get_ip_range_from_tokenizer(tokenizer_path, oov)
    print(f"Input range oov: {ip_range}")

    tokenizer_path = "./tokenizer_data/qwen2-math-7b-tokenizer.json"
    oov = False
    ip_range = get_ip_range_from_tokenizer(tokenizer_path, oov)
    print(f"Input range in-vocab: {ip_range}")

    oov = True
    ip_range = get_ip_range_from_tokenizer(tokenizer_path, oov)
    print(f"Input range oov: {ip_range}")
