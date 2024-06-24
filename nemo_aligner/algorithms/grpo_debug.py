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

from collections import defaultdict

import pandas as pd
import torch
from megatron.core import parallel_state
from megatron.core.utils import divide
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingRandomSampler
from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split
from nemo.utils import logging
from nemo_aligner.utils.distributed import (
    SyncTimer,
    pad_tensors_to_max_global_seq_len,
    pad_batch
)
from nemo_aligner.utils.ppo_utils import (
    calculate_kl_penalty,
    create_mask,
)

from nemo_aligner.utils.train_utils import clip_gradients
from nemo_aligner.utils.trainer_utils import check_progress, compute_num_steps_per_epoch
from nemo_aligner.utils.utils import clear_memory, cpu_dict


def compute_num_rollout_microbatches(dataloader):
    return divide(
        divide(dataloader.batch_sampler.global_batch_size, dataloader.batch_sampler.micro_batch_size),
        parallel_state.get_data_parallel_world_size(),
    )


class GRPODebugger:
    """Trainer to coordinate PPO training
    """

    def __init__(
        self,
        cfg: DictConfig,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        logger,
        ckpt_callback,
        run_timer,
        generation_iter,
        duplicate_prompts,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.ckpt_callback = ckpt_callback
        self.generation_iter = generation_iter
        self.duplicate_prompts = duplicate_prompts

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        # the step here is PPO step
        self.step = 0
        # keep track of how many times we optimized the actor
        self.ppo_optimization_step = 0

        # compute `max_steps`
        self.num_steps_per_epoch = compute_num_steps_per_epoch(self.train_dataloader.batch_sampler)
        self.set_max_steps()

        self.compute_init_policy_kl = self.cfg.initial_policy_kl_penalty > 0
        # size to pad our rollout batch to
        self.rollout_batch_seq_length = self.cfg.rollout_batch_seq_length

        # for wandb table
        self.train_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])
        self.val_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])

        self.timer = SyncTimer(
            reduction="mean", sync_cuda=True, buffer_size=1, reduce_op=torch.distributed.ReduceOp.MAX
        )

    def generate_ppo_data(self, rollout_batches):
        """generate ppo specific data for training
        """
        ppo_rollout_data = defaultdict(list)
        ppo_rollout_metrics = defaultdict(lambda: 0)
        num_samples = 0

        def post_process_tensor(tensor):
            return map(lambda x: x.flatten(), tensor.cpu().split(1, dim=0))

        for rollout_batch in rollout_batches:
            # NOTE: all items in rollout batch or out of this computation
            # must have a leading B dimension
            prompt_lengths = rollout_batch["prompt_lengths"]
            response_tokens = rollout_batch["response_tokens"]
            rewards = rollout_batch["rewards"] - self.cfg.initial_policy_kl_penalty * rollout_batch["init_policy_kl"]
            reward_mean = rollout_batch["reward_mean"]
            reward_std = rollout_batch["reward_std"]
            mask = rollout_batch["mask"]
            logprobs = rollout_batch["logprobs"]

            num_samples += prompt_lengths.size(0)

            # collect everything we need to train PPO
            ppo_rollout_data["mask"].extend(post_process_tensor(mask))
            ppo_rollout_data["response_tokens"].extend(post_process_tensor(response_tokens))
            ppo_rollout_data["rewards"].extend(post_process_tensor(rewards))
            ppo_rollout_data["reward_mean"].extend(post_process_tensor(reward_mean))
            ppo_rollout_data["reward_std"].extend(post_process_tensor(reward_std))
            ppo_rollout_data["prev_logprobs"].extend(post_process_tensor(logprobs))
            

            # compute metrics
            # NOTE: this metric is not accumulated globally so it will differ between DP ranks
            ppo_rollout_metrics["init_policy_kl"] += (
                rollout_batch["init_policy_kl"].sum().item() if self.compute_init_policy_kl else 0
            )

        # average across the samples for the non global metrics
        ppo_rollout_metrics = {k: v / num_samples for k, v in ppo_rollout_metrics.items()}

        for k in ppo_rollout_data:
            rollout_batch_seq_length = self.rollout_batch_seq_length
            pad_value = self.model.tokenizer.eos_id

            # all other tensors in the rollout batch
            # will be B x S -1 (because we don't predict anything for the last token)
            if k != "response_tokens":
                pad_value = 0
                if rollout_batch_seq_length is not None:
                    rollout_batch_seq_length -= 1

            ppo_rollout_data[k] = pad_tensors_to_max_global_seq_len(
                ppo_rollout_data[k],
                pad_value=pad_value,
                group=parallel_state.get_data_parallel_group(),
                sequence_length_to_pad_to=rollout_batch_seq_length,
            )

        return ppo_rollout_data, cpu_dict(ppo_rollout_metrics)
    
    def get_group_statistics(self, batch):
        '''
        Function to select the RLOO baseline for each (prompt, response) pair in the batch
        '''
        unique_prompts = torch.unique(batch["prompt_tokens"], dim=0)
        regularized_reward = batch["rewards"] - self.cfg.initial_policy_kl_penalty * batch["init_policy_kl"]

        batch["reward_mean"] = torch.zeros_like(batch["rewards"])
        batch["reward_std"] = torch.ones_like(batch["rewards"])
        for i in range(len(unique_prompts)):
            prompt_idx = torch.arange(len(batch["prompt_tokens"]))[(batch["prompt_tokens"] == unique_prompts[i]).all(1)]

            batch["reward_mean"][prompt_idx] = regularized_reward[prompt_idx].mean()
            batch["reward_std"][prompt_idx] = regularized_reward[prompt_idx].std() + 1e-6

        return batch
    
    def _run_inference(self, dataloader_iter, num_microbatches, is_validation):
        """this function is run per DP so the metrics need to be computed globally
        """
        rollout_batches = []
        if not is_validation:
            for _, inference_batch in zip(range(num_microbatches), dataloader_iter):

                current_batch = None
                inference_batch_duplicated = {
                    'text':torch.concatenate([inference_batch['text']] * self.duplicate_prompts, dim=0),
                    'length':torch.concatenate([inference_batch['length']] * self.duplicate_prompts, dim=0),
                    'attention_mask':inference_batch['attention_mask'],
                    'loss_mask':torch.concatenate([inference_batch['loss_mask']] * self.duplicate_prompts, dim=0),
                    'position_ids':torch.concatenate([inference_batch['position_ids']] * self.duplicate_prompts, dim=0),
                }
                for _ in range(self.generation_iter):
                    
                    if current_batch is None:
                        rollout_batch = self.model.infer(inference_batch_duplicated) # Note that critic mbs has to be set correctly
                        current_batch = rollout_batch
                        current_batch["prompt_tokens"] = inference_batch_duplicated["text"]
                        
                    else:
                        rollout_batch = self.model.infer(inference_batch_duplicated)
                        # Need to pad response tokens before concatenating
                        current_batch["response_tokens"], rollout_batch["response_tokens"] = pad_batch(current_batch["response_tokens"], rollout_batch["response_tokens"], self.model.tokenizer.eos_id)
                        current_batch["logprobs"], rollout_batch["logprobs"] = pad_batch(current_batch["logprobs"], rollout_batch["logprobs"], 0)

                        # Concat tensors
                        current_batch["response_tokens"] = torch.concatenate([current_batch["response_tokens"], rollout_batch["response_tokens"]], dim=0)
                        current_batch["logprobs"] = torch.concatenate([current_batch["logprobs"], rollout_batch["logprobs"]], dim=0)
                        current_batch["response_lengths"] = torch.concatenate([current_batch["response_lengths"], rollout_batch["response_lengths"]], dim=0)
                        current_batch["prompt_lengths"] = torch.concatenate([current_batch["prompt_lengths"], rollout_batch["prompt_lengths"]], dim=0)
                        current_batch["prompt_tokens"] = torch.concatenate([current_batch["prompt_tokens"], inference_batch_duplicated["text"]], dim=0)

                    # Get reward and init_policy logprobs
                    rewards = 1 / rollout_batch["response_lengths"].unsqueeze(-1) * 200 #self.rm_critic.infer_rm_critic(rollout_batch).result().detach()
                    init_policy_logprobs = self.model.get_init_policy_logprobs([rollout_batch])[0]
                    

                    if "rewards" in current_batch:
                        current_batch["rewards"] = torch.concatenate([current_batch["rewards"], rewards], dim=0)
                        current_batch["init_logprobs"], _ = pad_batch(current_batch["init_logprobs"], init_policy_logprobs, 0)
                        current_batch["init_logprobs"] = torch.concatenate([current_batch["init_logprobs"], init_policy_logprobs], dim=0)
                    else:
                        current_batch["rewards"] = rewards
                        current_batch["init_logprobs"] = init_policy_logprobs

                # Compute baselines and KL penalty here, as we need to use the inference batch in their computation

                if self.compute_init_policy_kl:
                    init_policy_kl = calculate_kl_penalty(
                        log_probs_a=current_batch["logprobs"],
                        log_probs_b=current_batch["init_logprobs"],
                        use_absolute_kl=self.cfg.use_absolute_kl,
                    )
                else:
                    init_policy_kl = torch.tensor(0, dtype=current_batch["logprobs"].dtype, device=current_batch["logprobs"].device)
                
                mask = create_mask(values=current_batch["logprobs"], prompt_lengths=current_batch["prompt_lengths"], response_lengths=current_batch["response_lengths"])
                current_batch["mask"] = mask
                current_batch["init_policy_kl"] = (init_policy_kl * mask).sum(-1).unsqueeze(-1)
            
                current_batch = self.get_group_statistics(current_batch)
                
                rollout_batches.append(current_batch)

        else:
            for _, inference_batch in zip(range(num_microbatches), dataloader_iter):
                rollout_batch = self.model.infer(inference_batch) # Here we meed to get the prompts as well
                
                rewards = 1 / rollout_batch["response_lengths"].unsqueeze(-1) * 200 #self.rm_critic.infer_rm_critic(rollout_batch).result().detach()
                rollout_batch["rewards"] = rewards.unsqueeze(-1)
                rollout_batches.append(rollout_batch)
            
        return rollout_batches, cpu_dict(self.compute_global_rollout_metrics(rollout_batches))

    def compute_global_rollout_metrics(self, rollout_batches):
        metrics = defaultdict(lambda: 0)
        table = {}

        num_samples = 0
        for i, rollout_batch in enumerate(rollout_batches):
            prompt_lengths = rollout_batch["prompt_lengths"]
            response_lengths = rollout_batch["response_lengths"]
            response_tokens = rollout_batch["response_tokens"]
            rewards = rollout_batch["rewards"]

            # table logging
            if i == 0:
                reward = rewards[0]
                prompt_length = prompt_lengths[0]
                response_length = response_lengths[0]
                response_token = response_tokens[0]

                table["reward"] = reward.item()
                table["prompt"] = self.model.tokenizer.ids_to_text(response_token[:prompt_length].tolist())
                table["response"] = self.model.tokenizer.ids_to_text(
                    response_token[prompt_length:response_length].tolist()
                )

            metrics["response_lengths"] += (response_lengths - prompt_lengths).sum()
            metrics["prompt_lengths"] += prompt_lengths.sum()
            metrics["rewards"] += rewards.sum()
            num_samples += prompt_lengths.size(0)

        tensor_to_accumulate = torch.tensor(
            [metrics["response_lengths"], metrics["prompt_lengths"], metrics["rewards"], num_samples],
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(tensor_to_accumulate, group=parallel_state.get_data_parallel_group())

        (
            global_response_lengths,
            global_prompt_lengths,
            global_rewards,
            global_num_samples,
        ) = tensor_to_accumulate.tolist()

        metrics = {
            "table": table,
            "global_response_lengths_mean": global_response_lengths / global_num_samples,
            "global_prompt_lengths": global_prompt_lengths / global_num_samples,
            "global_rewards": global_rewards / global_num_samples,
        }

        return metrics

    @torch.no_grad()
    def run_validation(self):
        self.model.prepare_for_inference()

        num_val_micro_batches = compute_num_rollout_microbatches(self.val_dataloader)
        val_dataloader = iter(self.val_dataloader)

        _, rollout_metrics = self._run_inference(val_dataloader, num_val_micro_batches, is_validation=True)
        self.model.finish_inference()
        return rollout_metrics

    @torch.no_grad()
    def generate_rollouts(self, dataloader_iter, num_microbatches):
        self.model.prepare_for_inference()

        rollout_batches, rollout_metrics = self._run_inference(dataloader_iter, num_microbatches, is_validation=False)

        ppo_rollout_data, ppo_rollout_metrics = map(cpu_dict, self.generate_ppo_data(rollout_batches))

        self.model.finish_inference()

        self.consumed_samples += (
            ppo_rollout_data["response_tokens"].size(0) * parallel_state.get_data_parallel_world_size()
        )
        return ppo_rollout_data, rollout_metrics | ppo_rollout_metrics | {"consumed_samples": self.consumed_samples}

    def run_training(self, dataloader_iter):
        self.model.prepare_for_training()

        for batch in dataloader_iter:
            self.timer.start("train_step_time")
            self.optimizer.zero_grad()

            self.model.prepare_for_training_step()
            loss_mean, metrics = self.model.get_loss_and_metrics(batch=batch, forward_only=False)
            self.model.finish_training_step()

            grad_norm = clip_gradients(self.model, self.cfg.gradient_clip_val)
            grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            lr = self.optimizer.param_groups[0]["lr"]

            self.optimizer.step()
            self.scheduler.step()

            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm

            metrics.update({"lr": lr, "loss": loss_mean, "optim_step": self.ppo_optimization_step})

            self.timer.stop("train_step_time")
            metrics["train_step_time"] = self.timer.get("train_step_time")

            self.logger.log_metrics(
                metrics, step=self.step, prefix="train_optim/",
            )

            self.ppo_optimization_step += 1

        self.model.finish_training()

        # zero grad again incase it frees up grad mem
        self.optimizer.zero_grad()
        return loss_mean, metrics

    def fit(self):
        if (not isinstance(self.train_dataloader.batch_sampler, MegatronPretrainingRandomSampler)) and (
            self.cfg.max_epochs is not None and self.cfg.max_epochs > 1
        ):
            # if you use MegatronPretrainingBatchSampler as the batch_sampler passed to your train dataloader (in builders.py)
            # then each epoch will repeat all your samples in the same order as the previous epoch, there is no shuffling
            # to fix this, you should use MegatronPretrainingRandomSampler instead, which alleviates this issue and allows
            # random shuffling for each epoch.
            raise ValueError(
                "max_epochs > 1 is not supported unless using `MegatronPretrainingRandomSampler` as the batch_sampler for your train dataloader"
            )

        epoch_iter = range(self.epoch, self.cfg.max_epochs)
        if len(epoch_iter) <= 0:
            # epoch done
            return

        for _ in epoch_iter:
            num_steps_in_epoch = min(
                self.max_steps - self.step, self.num_steps_per_epoch - self.step % self.num_steps_per_epoch
            )
            loop_iter = range(num_steps_in_epoch)

            if not loop_iter:
                return  # training ended

            dataloader_iter = iter(self.train_dataloader)

            global_pbar = tqdm(loop_iter, initial=self.step, total=self.max_steps, leave=True, desc="PPO Global Step")

            num_rollout_micro_batches = compute_num_rollout_microbatches(self.train_dataloader)
            dp_size = parallel_state.get_data_parallel_world_size()

            num_to_load_on_each_dp = divide(self.cfg.model_gbs, dp_size)

            self.run_timer.start_time()
            for _ in global_pbar:
                step_metrics = {}
                timing_metrics = {}

                self.timer.start("rollout_time")
                ppo_rollout_data, metrics = self.generate_rollouts(dataloader_iter, num_rollout_micro_batches)
                self.timer.stop("rollout_time")
                timing_metrics["rollout_time"] = self.timer.get("rollout_time")

                # logging
                table_metrics = metrics.pop("table")
                self.train_df.loc[len(self.train_df)] = [
                    self.step,
                    table_metrics["prompt"],
                    table_metrics["response"],
                    table_metrics["reward"],
                ]
                metrics["epoch"] = self.epoch + 1
                self.logger.log_metrics(
                    metrics, step=self.step, prefix="train_rollouts/",
                )
                self.logger.log_table(
                    key="table/train_rollouts", dataframe=self.train_df, step=self.step,
                )

                rollout_size = ppo_rollout_data["response_tokens"].size(0)
                rollout_dataloader_iter = get_iterator_k_split(
                    ppo_rollout_data, divide(rollout_size, num_to_load_on_each_dp)
                )
                # start training
                clear_memory()
                self.timer.start("train_time")
                self.run_training(rollout_dataloader_iter)
                self.timer.stop("train_time")
                timing_metrics["train_time"] = self.timer.get("train_time")

                self.step += 1

                run_time_exceeded = self.run_timer.is_finished()
                run_val, save_model, is_train_end = check_progress(
                    self.step,
                    self.max_steps,
                    self.cfg.val_check_interval,
                    self.cfg.save_interval,
                    1.0,  # TODO:(geshen): allow for limit val batches
                    run_time_exceeded=run_time_exceeded,
                )

                if run_val:
                    self.timer.start("validation_time")
                    val_metrics = self.run_validation()
                    self.timer.stop("validation_time")
                    timing_metrics["validation_time"] = self.timer.get("validation_time")

                    val_table_metrics = val_metrics.pop("table")

                    self.val_df.loc[len(self.val_df)] = [
                        self.step,
                        val_table_metrics["prompt"],
                        val_table_metrics["response"],
                        val_table_metrics["reward"],
                    ]
                    self.logger.log_metrics(val_metrics, step=self.step, prefix="val_rollouts/")
                    self.logger.log_table("table/val_rollouts", dataframe=self.val_df, step=self.step)

                    step_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                self.logger.log_metrics(timing_metrics, step=self.step, prefix="timers/")

                step_metrics.update(timing_metrics)
                step_metrics.update({f"train_{k}": v for k, v in metrics.items()})
                global_pbar.set_postfix(step_metrics)

                if save_model:
                    step_metrics = {k: torch.as_tensor(v) for k, v in step_metrics.items()}
                    self.save(step_metrics, is_train_end=is_train_end)

                if run_time_exceeded:
                    logging.info(f"Time limit given by run_timer={self.run_timer} reached. Stopping run")
                    return

        self.logger.finalize()

    def state_dict(self):
        return {
            "step": self.step,
            "consumed_samples": self.consumed_samples,
            "epoch": self.epoch,
            "ppo_optimization_step": self.ppo_optimization_step,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.consumed_samples = state_dict["consumed_samples"]
        self.ppo_optimization_step = state_dict["ppo_optimization_step"]

        loaded_values = [self.step, self.consumed_samples, self.ppo_optimization_step]

        # make sure everyone loaded the same checkpoint as rank 0
        to_broadcast = torch.tensor(loaded_values, dtype=torch.float32, device=torch.cuda.current_device())
        torch.distributed.broadcast(to_broadcast, 0)

        assert loaded_values == to_broadcast.tolist()
        # restore max steps we need to run for
        self.set_max_steps()

    def save(self, extra_candidates=None, is_train_end=False):
        self.model.prepare_for_training()
        # load back in the adam states if needed
        torch.cuda.synchronize()
        torch.distributed.barrier()

        if extra_candidates is None:
            extra_candidates = {}

        monitor_candidates = {k: torch.tensor(v, dtype=torch.int32) for k, v in self.state_dict().items()}
        monitor_candidates.update(extra_candidates)


        self.ckpt_callback.custom_save(monitor_candidates=monitor_candidates, is_train_end=is_train_end)

        self.model.finish_training()

    def set_max_steps(self):
        self.max_steps = self.num_steps_per_epoch * self.cfg.max_epochs

        if (max_steps := self.cfg.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.step // self.num_steps_per_epoch
