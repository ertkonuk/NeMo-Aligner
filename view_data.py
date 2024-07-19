import json
from instruction_following_eval.evaluation_main import InputExample, test_instruction_following_strict, test_instruction_following_loose

# with open("data/extra_id_prefix_end_with_backslash_n_extra_id_1_jsonl/scale_2_filtered_v2_val_prompts.jsonl", "r") as f:
#     data = [json.loads(line) for line in f]


with open("data/ifeval_preferences.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

correct = 0
for datapoint in data:
    if datapoint["args"]["instruction_kwargs"][0] is None:
        datapoint["args"]["instruction_kwargs"] = [{}]
    example = InputExample(
        key="",
        instruction_id_list=datapoint["args"]["instruction_id_list"],
        prompt=datapoint["prompt"],
        kwargs=datapoint["args"]["instruction_kwargs"]
    )

    output = test_instruction_following_strict(example, {datapoint["prompt"]:datapoint["chosen_response"]})
    print(output.follow_instruction_list)
    correct += int(output.follow_instruction_list[0])

print(correct/ len(data))