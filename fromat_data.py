import json

with open("data/ifeval_train_prompts.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        data["text"] = None
        exit()