import json
from pprint import pprint

# with open("data/extra_id_prefix_end_with_backslash_n_extra_id_1_jsonl/scale_2_filtered_v2_train_prompts.jsonl", "r") as f:
#     data = [json.loads(line) for line in f]

with open("data/ifeval_preferences.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
print("\n\n\n\n")
pprint(data[0])
print("\n\n\n\n")
print(data[0]['chosen_response'])
print("-"*30)
print(data[0]['rejected_response'])
print(data[0].keys())