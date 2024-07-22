import json

with open("data/preference.rm9200.reject-rand-chosen-min-0.0.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        print(data)
        exit()