# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from omegaconf import DictConfig
from concurrent import futures

import torch

from nemo_aligner.experimental.grpo.utils import parallel_state
from nemo_aligner.utils.utils import masked_mean
from nemo_aligner.experimental.grpo.experience.interfaces import EnvironmentInterface
from nemo_aligner.experimental.grpo.experience.environments.metrics import calculate_pass_rate_per_prompt
from nemo_aligner.servers.http_communicator import FlaskCommunicator
from nemo_aligner.experimental.grpo.experience.environments.format_checker import FormatChecker

class IFEvalEnvironment(EnvironmentInterface):
    def __init__(self, cfg: DictConfig):
        self.executor = futures.ThreadPoolExecutor()
        self.communicator = FlaskCommunicator(cfg.servers)
        
        print(f"Started IfevalEnvironment client with {cfg.servers}")
        
    def start_step(self, interactions, metadata, is_end):
        """
        metadata: List[Dict]. Needs to contain a "ground_truth" key, which is what
                              the grader will use to evaluate correctness.
        """
        if parallel_state.is_model_parallel_src_rank():
            # fold all interactions after the prompt together
            prompts = [interaction[0] for interaction in interactions]
            full_responses = [''.join(interaction[1:]) for interaction in interactions]

            print("--------------------------------")
            print(f"prompts: {prompts}")
            print("--------------------------------")
            responses = [interaction[-1] for interaction in interactions]
            responses = [r.split("</think>")[-1].strip() for r in responses]
            print("********************************")
            print(f"responses: {responses}")
            print("********************************")
            args = [g["args"] for g in metadata]

            print("### LEN OF PROMPTS", len(prompts))
            print("### LEN OF FULL RESPONSES", len(full_responses))
            print("### LEN OF IS END", len(is_end))
            format_rewards = FormatChecker.calculate_format_metrics(prompts, full_responses, is_end)
            print("### FORMAT REWARDS length", len(format_rewards))

            data = {
                "pred_responses": responses,
                "args": args,
                "prompts": prompts,
                "format_rewards": format_rewards,
            }
            print(f"data: {data}")
            return self.communicator.send_data_to_server("ifeval_grader", data)
        return None

    def finish_step(self, future):
        # gets the future result and also broadcasts within the current MP group
        results = self.communicator.get_result(future, "rewards")

        th_rewards = torch.tensor(results).squeeze(1)
        print('th rewards shape', th_rewards.shape)
        return None, None, th_rewards, torch.ones(th_rewards.shape[0],)
    
    def global_post_process_and_metrics(self, batch):
        """
        Computes metrics for this environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed 
        calculations if you'd prefer for heavy metrics. 
        """
        pre_is_end_reward = batch["rewards"].mean().item()
        batch["rewards"] = batch["rewards"] * batch["is_end"] # set a reward of 0 for any incorrectly ended sequences

        # print("### WHAT IS IS_END", batch["is_end"])
        # assert torch.all(0 <= batch["is_end"] <= 1)

        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["response_lengths"] - batch["prompt_lengths"])[batch["rewards"] == 1].float().mean().item()
            )
        else:
            correct_solution_generation_lengths = 0

        format_rewards = FormatChecker.calculate_format_metrics(
            batch["prompt_sentences"],
            batch["response_sentences"],
            batch["is_end"]
        )
        
        metrics = {
            #"table": table, TODO @sahilj WIP
            "pre_is_end_reward": pre_is_end_reward,
            "is_end_sum": batch["is_end"].float().sum().item(),
            "is_end_mean": batch["is_end"].float().mean().item(),
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(batch["text"], batch["rewards"]),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "response_lengths": batch["response_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "generation_lengths": (batch["response_lengths"] - batch["prompt_lengths"]).float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
            "if_eval_format_rewards": torch.tensor(format_rewards).float().mean().item(),
        }
        
        return batch, metrics
