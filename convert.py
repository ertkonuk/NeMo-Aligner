import json
import random

with open("data/ifeval_preferences.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

for obj in data:
    obj["text"] = obj.pop("prompt")
    print(obj)
    if obj["args"]["instruction_kwargs"][0] is None:
        obj["args"]["instruction_kwargs"] = [{}]

random.seed(0)
random.shuffle(data)

with open("data/ifeval_train_prompts.jsonl", "w") as f:
    for entry in data[512:]:
        f.write(json.dumps(entry) + "\n")


with open("data/ifeval_val_prompts.jsonl", "w") as f:
    for entry in data[:512]:
        f.write(json.dumps(entry) + "\n")