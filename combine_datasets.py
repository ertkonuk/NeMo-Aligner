import json
import random

preference = []
with open("data/preference.rm9200.reject-rand-chosen-min-0.0.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        text = data["prompt"]
        data["text"] = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""".format(text=text)
        data["args"] = {"task":"reward_model"}
        preference.append(data)

ifeval = []
with open("data/ifeval_preferences.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        text = data["prompt"]
        data["text"] = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    <|eot_id|><|start_header_id|>user<|end_header_id|>

    {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """.format(text=text)
        data["args"]["task"] = "ifeval"
        if data["args"]["instruction_kwargs"][0] is None:
            data["args"]["instruction_kwargs"] = [{}]
        ifeval.append(data)



random.seed(0)
random.shuffle(preference)
random.shuffle(ifeval)

train = preference[128:] + ifeval[128:]
val = preference[:128] + ifeval[:128]

with open("data/llama3_combined_train_data.jsonl", "w") as f:
    for data in train:
        jsonline = json.dumps(data)
        f.write(jsonline + "\n")

with open("data/llama3_combined_val_data.jsonl", "w") as f:
    for data in val:
        jsonline = json.dumps(data)
        f.write(jsonline + "\n")

with open("data/llama3_ifeval_train_data.jsonl", "w") as f:
    for data in ifeval[128:]:
        jsonline = json.dumps(data)
        f.write(jsonline + "\n")

with open("data/llama3_ifeval_val_data.jsonl", "w") as f:
    for data in ifeval[:128]:
        jsonline = json.dumps(data)
        f.write(jsonline + "\n")

with open("data/llama3_chat_train_data.jsonl", "w") as f:
    for data in preference[128:]:
        jsonline = json.dumps(data)
        f.write(jsonline + "\n")

with open("data/llama3_chat_val_data.jsonl", "w") as f:
    for data in preference[:128]:
        jsonline = json.dumps(data)
        f.write(jsonline + "\n")




print(len(train), len(val))