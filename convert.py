import json
import random

with open("data/ifeval_preferences.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

for obj in data:
    text = obj["prompt"]
    obj["text"] = f'<extra_id_0>System\n\n<extra_id_1>User\n{text}\n<extra_id_1>Assistant\n'
    if obj["args"]["instruction_kwargs"][0] is None:
        obj["args"]["instruction_kwargs"] = [{}]

random.seed(0)
random.shuffle(data)

with open("data/ifeval_train_prompts.jsonl", "w") as f:
    for entry in data[128:]:
        f.write(json.dumps(entry) + "\n")


with open("data/ifeval_val_prompts.jsonl", "w") as f:
    for entry in data[:128]:
        f.write(json.dumps(entry) + "\n")