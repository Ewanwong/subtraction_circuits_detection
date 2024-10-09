# import sglang as sgl
import openai

# @sgl.function
# def math_complete(s, prompt):
#     params = get_default_prompting_params()
#     s += prompt + sgl.gen("answer", **params)


def get_default_prompting_params():
    return {
        "max_tokens": 100,
        "temperature": 0.0,
        "top_p": 1,
        "top_k": 1,
        "stop": "<eop>",
        "return_text_in_logprobs": True,
        "return_logprob": True,
    }


def openai_complete(prompt, client):
    model = client.models.list()
    response = client.completions.create(
        model=model.data[0].id,
        # model="allenai/OLMo-7B-0724-Instruct-hf",
        prompt=prompt,
        temperature=0,
        max_tokens=10,
        stop="}"
    )
    return response


if __name__ == "__main__":
    port = 12323
    # sgl.set_default_backend(sgl.RuntimeEndpoint(f"http://localhost:{port}", api_key="sk_noreq"))
    client = openai.Client(base_url="http://127.0.0.1:8000/v1", api_key="sk_noreq")
    msgs = [
        {"prompt": "### 796 + 593 = answer{1389} ### <eop>\n### 3 + 705 = answer{708} ### <eop>\n### 3 + 526 = answer{529} ### <eop>\n### 241 + 996 = answer{1237} ### <eop>\n### 829 + 121 = answer{950} ### <eop>\n### 23 + 368 = answer{"},
        {"prompt": "### 339 + 752 = answer{1091} ### <eop>\n### 436 + 249 = answer{685} ### <eop>\n### 62 + 118 = answer{180} ### <eop>\n### 172 + 246 = answer{418} ### <eop>\n### 1 + 207 = answer{208} ### <eop>\n### 743 + 427 = answer{"},
        {"prompt": "### 380 + 995 = answer{1375} ### <eop>\n### 845 + 425 = answer{1270} ### <eop>\n### 560 + 501 = answer{1061} ### <eop>\n### 819 + 214 = answer{1033} ### <eop>\n### 629 + 424 = answer{1053} ### <eop>\n### 425 + 441 = answer{"},
        {"prompt": "### 90 + 862 = answer{952} ### <eop>\n### 127 + 51 = answer{178} ### <eop>\n### 125 + 522 = answer{647} ### <eop>\n### 219 + 492 = answer{711} ### <eop>\n### 347 + 846 = answer{1193} ### <eop>\n### 790 + 679 = answer{"},
    ]
    # states = math_complete.run_batch(
    #     msgs,
    #     progress_bar=True,
    # )
    # for s in states:
    #     print(s.text())
    #     meta_info = s.get_meta_info("answer")
    #     print(meta_info['completion_tokens'])
    #     print(meta_info['output_token_logprobs'])

    for msg in msgs:
        resp = openai_complete(msg["prompt"], client)
        print(resp.choices[0].text)
