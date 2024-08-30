from datasets import load_dataset
import json
import numpy as np
import re
import random


ds = load_dataset("codeparrot/apps", split="train")
iterator = iter(ds)
np.random.seed(0)
data = []
for i, x in enumerate(iterator):
    if "fn_name" in x["input_output"]:
        # Format = mbpp
        #"\"\"\"\nWrite a function to check if each element of second tuple is smaller than its corresponding element in the first tuple.\nassert check_smaller((1, 2, 3), (2, 3, 4)) == False\n\"\"\"\n"
        solution = json.loads(x["solutions"])[0]
        question = x["question"]
        try:
            input_output = json.loads(x["input_output"])
        except:
            continue
        if len(input_output["inputs"]) == 0:
            continue
        
        fn_name = input_output["fn_name"]
        if fn_name == "__init__":
            continue
        question += f"\nThe function should be called {fn_name}."
        text = f"""You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
Here is the given problem and test examples:
{question}
Please use the python programming language to solve this problem.
Please return only the function in one code block. You do not need to run the code for the test cases.
This code block should be in the following format:
```python
# Your codes here
```"""
        prompt = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

        obj = {"text":prompt,
        "args":{
            "task":"coding",
            "inputs":input_output["inputs"],
            "outputs":input_output["outputs"],
            "fn_name":fn_name
        }}
        data.append(obj)

print(len(data))
random.seed(0)
random.shuffle(data)

val = data[:512]
train = data[512:]

with open("../data/llama3_apps_short_train.jsonl", "w") as f:
    for data in train:
        jsonline = json.dumps(data)
        f.write(jsonline + "\n")

with open("../data/llama3_apps_short_val.jsonl", "w") as f:
    for data in val:
        jsonline = json.dumps(data)
        f.write(jsonline + "\n")