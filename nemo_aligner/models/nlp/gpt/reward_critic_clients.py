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
from code_eval.test_single import unsafe_execute, execute_code
from multiprocessing import Array, Value
import multiprocessing
import time

"""A remote client that acts like a real Reward Model and Critic forwards all requests from the actor
    over to the remote PyTrition server
"""


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
class RemoteGPTRMCriticClient:
    cfg: DictConfig

    def __post_init__(self):
        cfg = self.cfg

        critic_ip_and_port = (cfg.critic.ip, cfg.critic.port)
        server_dict = {
            cfg.critic.name.train: critic_ip_and_port,
            cfg.critic.name.infer: critic_ip_and_port,
            cfg.critic.name.save: critic_ip_and_port,
        }

        if not cfg.combine_rm_and_critic_server:
            server_dict[cfg.reward_model.name] = (cfg.reward_model.ip, cfg.reward_model.port)

        self.communicator = HTTPCommunicator.create_http_communicator_from_dict(server_dict)
        self.communicator.print_server_dict()
        self.combine_rm_and_critic_server = self.cfg.combine_rm_and_critic_server
        self.pad_to_length = self.cfg.pad_to_length

    def infer_rm_critic(self, rollout_batch, model, args):
        response_tokens = rollout_batch["response_tokens"].cpu()
        og_seq_length = response_tokens.size(-1)

        

        if self.pad_to_length is not None:
            assert (
                og_seq_length <= self.pad_to_length
            ), f"original shape before padding {og_seq_length} is higher than {self.pad_to_length}"
            response_tokens = torch.nn.functional.pad(
                response_tokens, (0, self.pad_to_length - response_tokens.size(-1)), value=0
            )

        send_data = {
            "tokens": response_tokens.numpy(),
            "sequence_lengths": rollout_batch["response_lengths"].unsqueeze(1).cpu().numpy(),
        }

        critic_future = run_if_model_parallel_src(
            self.communicator.send_data_to_server, server_name=self.cfg.critic.name.infer, data=send_data,
        )

        rm_future = None
        if not self.combine_rm_and_critic_server:
            rm_future = run_if_model_parallel_src(
                self.communicator.send_data_to_server, server_name=self.cfg.reward_model.name, data=send_data,
            )

        return RMCriticFutureResult(critic_future, rm_future, self.combine_rm_and_critic_server, og_seq_length)

    def train(self, ppo_rollout_data):
        send_data = {}

        func = partial(
            gather_tensor,
            dst=parallel_state.get_data_parallel_src_rank(),
            group=parallel_state.get_data_parallel_group(),
        )

        send_data["tokens"] = func(ppo_rollout_data["response_tokens"], dtype=torch.int64)
        send_data["returns"] = func(ppo_rollout_data["returns"], dtype=torch.float32)
        send_data["prev_values"] = func(ppo_rollout_data["values"], dtype=torch.float32)
        send_data["mask"] = func(ppo_rollout_data["mask"], dtype=torch.float32)

        future = None
        if torch.distributed.get_rank() == 0:
            send_data = {k: torch.cat(v, dim=0).detach().cpu().numpy() for k, v in send_data.items()}
            future = self.communicator.send_data_to_server(
                server_name=self.cfg.critic.name.train, data=send_data, batching=False
            )

        return future

    def save(self):
        save_future = None
        if torch.distributed.get_rank() == 0:
            send_data = {"dummy_var": np.array([0])}
            save_future = self.communicator.send_data_to_server(
                server_name=self.cfg.critic.name.save, data=send_data, batching=False
            )

        return SaveFuture(save_future)


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

    def infer_rm_critic(self, rollout_batch):
        response_tokens = rollout_batch["response_tokens"].cpu()
        
        og_seq_length = response_tokens.size(-1)

        if self.pad_to_length is not None:
            assert (
                og_seq_length <= self.pad_to_length
            ), f"original shape before padding {og_seq_length} is higher than {self.pad_to_length}"
            response_tokens = torch.nn.functional.pad(
                response_tokens, (0, self.pad_to_length - response_tokens.size(-1)), value=0
            )

        send_data = {
            "tokens": response_tokens.numpy(),
            "sequence_lengths": rollout_batch["response_lengths"].unsqueeze(1).cpu().numpy(),
        }

        rm_future = run_if_model_parallel_src(
            self.communicator.send_data_to_server, server_name=self.cfg.reward_model.name, data=send_data
        )

        return RMFutureResult(rm_future)


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

    # def coding_rewards(self, prompt, response, args):
    #     fn_name = args["fn_name"]
    #     inputs = args["inputs"]
    #     outputs = args["outputs"]
    #     progress = 0
    #     stat = 0
    #     # details = [False for _ in range(len(inputs))]
    #     details = Array("b", [False for _ in range(len(inputs))])
    #     time_limits = [5 for _ in range(len(inputs))]

    #     try:
    #         code = response.split("```python\n")[1].split("```")[0].split("assert")[0].split("# Test")[0].split("# Unit")[0].strip()
    #     except:
    #         code = response.replace("# Your codes here\n", "").split("```")[0].strip()


    #     _, results = execute_code(entry_point=fn_name, code=code, inputs=inputs, expected=outputs, time_limits=time_limits, atol=1e-6, stat=stat, details=details, progress=progress)
    #     return int(all(results))

    
    def task_mask(self, args, device):
        mask = torch.tensor([1 if arg["task"] in ["ifeval", "gsm8k", "coding"] else 0 for arg in args], device=device).float()
        return mask.unsqueeze(-1)

    def infer_rm_critic(self, rollout_batch, model, args):
        response_tokens = rollout_batch["response_tokens"].cpu()
        og_seq_length = response_tokens.size(-1)

        ifeval_rewards = []
        for i in range(rollout_batch["response_tokens"].size(0)):
            prompt = model.tokenizer.ids_to_text(rollout_batch["response_tokens"][i, :rollout_batch["prompt_lengths"][i]].tolist())
            response = model.tokenizer.ids_to_text(rollout_batch["response_tokens"][i, rollout_batch["prompt_lengths"][i]:rollout_batch["response_lengths"][i]].tolist())
            for end_string in self.cfg.end_strings:
                response = response.replace(end_string, "")
            ifeval_rewards.append(self.task_reward(prompt, response, args[i]))
            
        ifeval_mask = self.task_mask(args, device=rollout_batch["logprobs"].device)
        ifeval_rewards = torch.tensor(ifeval_rewards, device=rollout_batch["logprobs"].device).unsqueeze(-1)

        if self.pad_to_length is not None:
            assert (
                og_seq_length <= self.pad_to_length
            ), f"original shape before padding {og_seq_length} is higher than {self.pad_to_length}"
            response_tokens = torch.nn.functional.pad(
                response_tokens, (0, self.pad_to_length - response_tokens.size(-1)), value=0
            )

        send_data = {
            "tokens": response_tokens.numpy(),
            "sequence_lengths": rollout_batch["response_lengths"].unsqueeze(1).cpu().numpy(),
        }

        rm_future = run_if_model_parallel_src(
            self.communicator.send_data_to_server, server_name=self.cfg.reward_model.name, data=send_data
        )

        return RMFutureResult(rm_future), ifeval_rewards, ifeval_mask



@dataclass
class CodeEvaluator:
    cfg: DictConfig

    def __post_init__(self):
        cfg = self.cfg
    
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

    def coding_rewards(self, prompt, response, args):
        fn_name = args["fn_name"]
        inputs = args["inputs"]
        if len(inputs) == 0:
            return 0
        outputs = args["outputs"]
        progress = 0
        stat = 0
        details = Array("b", [False for _ in range(len(inputs))])
        time_limits = [5 for _ in range(len(inputs))]

        try:
            code = response.split("```python\n")[1].split("```")[0].split("assert")[0].split("# Test")[0].split("# Unit")[0].strip()
        except:
            code = response.replace("# Your codes here\n", "").split("```")[0].strip()

        p = multiprocessing.Process(
        target=execute_code,
        args=(
            fn_name,
            code,
            inputs,
            outputs,
            time_limits,
            1e-6,
            stat,
            details,
            progress,
        ),
        )
        p.start()
        timeout = sum(time_limits)
        p.join(timeout=timeout + 1)
        if p.is_alive():
            p.terminate()
            time.sleep(0.1)
        if p.is_alive():
            p.kill()
            time.sleep(0.1)

        print(stat, "!!!!!!!!!")

        print(len(inputs))
        return int(all(details))

    
    def task_mask(self, args, device):
        mask = torch.tensor([1 if arg["task"] in ["ifeval", "gsm8k", "coding"] else 0 for arg in args], device=device).float()
        return mask.unsqueeze(-1)

    def infer(self, rollout_batch, model, args):
        response_tokens = rollout_batch["response_tokens"].cpu()
        og_seq_length = response_tokens.size(-1)

        ifeval_rewards = []
        for i in range(rollout_batch["response_tokens"].size(0)):
            prompt = model.tokenizer.ids_to_text(rollout_batch["response_tokens"][i, :rollout_batch["prompt_lengths"][i]].tolist())
            response = model.tokenizer.ids_to_text(rollout_batch["response_tokens"][i, rollout_batch["prompt_lengths"][i]:rollout_batch["response_lengths"][i]].tolist())
            for end_string in self.cfg.end_strings:
                response = response.replace(end_string, "")
            ifeval_rewards.append(self.task_reward(prompt, response, args[i]))
            
        ifeval_mask = self.task_mask(args, device=rollout_batch["logprobs"].device)
        ifeval_rewards = torch.tensor(ifeval_rewards, device=rollout_batch["logprobs"].device).unsqueeze(-1)


        return ifeval_rewards, ifeval_mask