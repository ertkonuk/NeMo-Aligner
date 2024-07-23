import json
import random

datalist = []
with open("data/preference.rm9200.reject-rand-chosen-min-0.0.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        text = data["prompt"]
        data["text"] = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""".format(text=text)
        data["args"] = {"task":"reward_model"}
        datalist.append(data)


random.seed(0)
random.shuffle(datalist)

with open("data/llama3_train_data.jsonl", "w") as f:
    for data in datalist[128:]:
        jsonline = json.dumps(data)
        f.write(jsonline + "\n")

with open("data/llama3_val_data.jsonl", "w") as f:
    for data in datalist[:128]:
        jsonline = json.dumps(data)
        f.write(jsonline + "\n")