# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from omegaconf import DictConfig

from nemo_aligner.servers.http_communicator import HTTPCommunicator
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_mp, gather_tensor, run_if_model_parallel_src
from nemo_aligner.utils.server_utils import FutureResult
from instruction_following_eval.evaluation_main import InputExample, test_instruction_following_strict
import re
# from code_eval.test_single import unsafe_execute, execute_code
# from multiprocessing import Array, Value
# import multiprocessing
import time

"""A remote client that acts like a real Reward Model and Critic forwards all requests from the actor
    over to the remote PyTrition server
"""


class HelpsteerTemplate:
    def get_first_turn_template(self, text):
        return f"""<extra_id_0>System\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
<extra_id_1>User\n{text}"""

    def get_assistant_turn_template(self, text):
        return f"""\n<extra_id_1>Assistant\n{text}"""

    def get_user_turn_template(self, text):
        return f"""\n<extra_id_1>User\n{text}"""

    def add_ending(self, text):
        return f"""{text}\n<extra_id_2>"""



def chat_template(user_text, assistant_text, template):
    formatter = HelpsteerTemplate()
    
    text = ""
    for i in range(len(user_text)):
        if i == 0:
            text += formatter.get_first_turn_template(user_text[i])
        else:
            text += formatter.get_user_turn_template(user_text[i])
        text += formatter.get_assistant_turn_template(assistant_text[i])
    text = formatter.add_ending(text)
    return text


def extract_dialogue_helpsteer(text):
    user_pattern = r'<extra_id_1>User\n(.*?)\n<extra_id_1>'
    assistant_pattern = r'<extra_id_1>Assistant\n(.*?)\n(?:<extra_id_1>|<extra_id_2>)'
    
    user_text = re.findall(user_pattern, text, re.DOTALL)
    assistant_text = re.findall(assistant_pattern, text, re.DOTALL)
    
    return user_text, assistant_text

def extract_dialogue_llama(text):
    user_pattern = r'<\|eot_id\|><\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|start_header_id\|>'
    assistant_pattern = r'<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|start_header_id\|>'
    
    user_text = re.findall(user_pattern, text, re.DOTALL)
    assistant_text = re.findall(assistant_pattern, text, re.DOTALL)
    
    return user_text, assistant_text


def get_future_result(future, *keys):
    """It waits for the result of the future to be ready, gets the value with the given key,
    and broadcasts it to the model parallel group. Then it returns it as output.
    """
    output = None if future is None else future.result()

    results = []

    for key in keys:

        result = None
        if output is not None:
            result = torch.tensor(output[key], device=torch.cuda.current_device())

        ten = broadcast_2d_tensor_within_mp(result)

        results.append(ten)

    if len(results) == 1:
        return results[0]

    return results


class RMCriticFutureResult(FutureResult):
    def __init__(self, critic_future, rm_future, combine_rm_and_critic_server, og_seq_length):
        self.critic_future = critic_future
        self.rm_future = rm_future
        self.combine_rm_and_critic_server = combine_rm_and_critic_server

        self.og_seq_length = og_seq_length

    def result(self):
        if self.combine_rm_and_critic_server:
            rewards, values = get_future_result(self.critic_future, "rewards", "values")
        else:
            rewards = get_future_result(self.rm_future, "rewards")
            values = get_future_result(self.critic_future, "values")

        values = values[:, : self.og_seq_length - 1].contiguous()

        self.critic_future = None
        self.rm_future = None
        return rewards.flatten(), values

class RMFutureResult(FutureResult):
    def __init__(self, rm_future):
        self.rm_future = rm_future

    def result(self):
        rewards = get_future_result(self.rm_future, "rewards")

        return rewards
    


class SaveFuture(FutureResult):
    def __init__(self, pytriton_save_future):
        self.pytriton_save_future = pytriton_save_future

    def result(self):
        if self.pytriton_save_future is not None:
            self.pytriton_save_future.result()

        # need to make sure it's saved
        torch.distributed.barrier()


@dataclass
class RemoteGPTRMClient:
    cfg: DictConfig

    def __post_init__(self):
        cfg = self.cfg

        server_dict = {cfg.reward_model.name: (cfg.reward_model.ip, cfg.reward_model.port)}

        self.communicator = HTTPCommunicator.create_http_communicator_from_dict(server_dict)
        self.communicator.print_server_dict()
        self.combine_rm_and_critic_server = self.cfg.combine_rm_and_critic_server
        self.pad_to_length = self.cfg.pad_to_length

    def infer_rm_critic(self, rollout_batch, model):
        response_tokens = rollout_batch["response_tokens"].cpu()
        
        og_seq_length = response_tokens.size(-1)

        texts = []
        for i in range(rollout_batch["response_tokens"].size(0)):
            text = model.tokenizer.ids_to_text(rollout_batch["response_tokens"][i, :rollout_batch["response_lengths"][i]].tolist())
            user_text, assistant_text = extract_dialogue_llama(text + "<|start_header_id|>")
            user_text = [x.replace("<|eot_id|>", "") for x in user_text]
            assistant_text = [x.replace("<|eot_id|>", "") for x in assistant_text]
            print(text + "<|start_header_id|>")
            print("--"*80)
            print("USER TEXT", user_text)
            print("ASSISTANT_TEXT", assistant_text)
            text = chat_template(user_text=user_text, assistant_text=assistant_text, template="HS2")
            print("**"*80)
            print(text)
            print("0O0"*60)
            texts.append(text)

        send_data = {
            "sentences": _str_list2numpy(texts),
            }


        rm_future = run_if_model_parallel_src(
            self.communicator.send_data_to_server, server_name=self.cfg.reward_model.name, data=send_data
        )

        return RMFutureResult(rm_future)

def _str_list2numpy(str_list) -> np.ndarray:
    str_ndarray = np.array(str_list)[..., np.newaxis]
    return np.char.encode(str_ndarray, "utf-8")

@dataclass
class RemoteGPTMultitaskClient:
    cfg: DictConfig

    def __post_init__(self):
        cfg = self.cfg

        server_dict = {cfg.reward_model.name: (cfg.reward_model.ip, cfg.reward_model.port)}

        self.communicator = HTTPCommunicator.create_http_communicator_from_dict(server_dict)
        self.communicator.print_server_dict()
        self.combine_rm_and_critic_server = self.cfg.combine_rm_and_critic_server
        self.pad_to_length = self.cfg.pad_to_length
    
    def ifeval_rewards(self, prompt, response, args):
        
        example = InputExample(
            key="",
            instruction_id_list=args["instruction_id_list"],
            prompt=prompt,
            kwargs=args["instruction_kwargs"]
        )
        try:
            output = test_instruction_following_strict(example, {prompt:response})
        except:
            output = [False]
        # queue.put(float(all(output.follow_instruction_list)))
        return float(all(output.follow_instruction_list))
    
    def task_reward(self, prompt, response, args):
        if args["task"] == "ifeval":
            return self.ifeval_rewards(prompt, response, args)
        elif args["task"] == "gsm8k":
            return self.gsm8k_rewards(prompt, response, args)
        elif args["task"] == "coding":
            return self.coding_rewards(prompt, response, args)
        else:
            return 0
    
    def gsm8k_rewards(self, prompt, response, args):
        ans = args["answer"]
        pattern = r"-?\$?\d[\d,]*\.?\d*|-?\.\d+"
        matches = re.findall(pattern, response)
        # print(prompt, response, matches, ans)
        if matches:
            try:
                prediction = float(matches[-1].replace('$', '').replace(',', ''))
                return int(prediction == ans)
            except:
                return 0
        else:
            return 0

    
    def task_mask(self, args, device):
        # mask = torch.tensor([1 if arg["task"] in ["ifeval", "gsm8k", "coding"] else 0 for arg in args], device=device).float()
        mask = torch.tensor([1 if arg["task"] in ["ifeval"] else 0 for arg in args], device=device).float()
        return mask.unsqueeze(-1)

    def infer_rm_critic(self, rollout_batch, model, args):
        response_tokens = rollout_batch["response_tokens"].cpu()
        og_seq_length = response_tokens.size(-1)


        # Calculate task rewards
        ifeval_rewards = []
        for i in range(rollout_batch["response_tokens"].size(0)):
            prompt = model.tokenizer.ids_to_text(rollout_batch["response_tokens"][i, :rollout_batch["prompt_lengths"][i]].tolist())
            response = model.tokenizer.ids_to_text(rollout_batch["response_tokens"][i, rollout_batch["prompt_lengths"][i]:rollout_batch["response_lengths"][i]].tolist())
            for end_string in self.cfg.end_strings:
                response = response.replace(end_string, "")
            ifeval_rewards.append(self.task_reward(prompt, response, args[i]))
            
        ifeval_mask = self.task_mask(args, device=rollout_batch["logprobs"].device)
        ifeval_rewards = torch.tensor(ifeval_rewards, device=rollout_batch["logprobs"].device).unsqueeze(-1)

        # Calculate rm rewards (needs reformatting)
        texts = []
        for i in range(rollout_batch["response_tokens"].size(0)):
            text = model.tokenizer.ids_to_text(rollout_batch["response_tokens"][i, :rollout_batch["response_lengths"][i]].tolist())
            if ifeval_mask[i]:
                text = "\n<extra_id_2>"
                texts.append(text)
                continue
            user_text, assistant_text = extract_dialogue_llama(text + "<\|eot_id\|>")
            text = chat_template(user_text=user_text, assistant_text=assistant_text, template="HS2")
            print(text)
            print("--"*80)
            print("USER TEXT", user_text)
            print("ASSISTANT_TEXT", assistant_text)
            print("-*"*80)
            texts.append(text)

        send_data = {
            "sentences": _str_list2numpy(texts),
            }


        rm_future = run_if_model_parallel_src(
            self.communicator.send_data_to_server, server_name=self.cfg.reward_model.name, data=send_data
        )

        return RMFutureResult(rm_future), ifeval_rewards, ifeval_mask