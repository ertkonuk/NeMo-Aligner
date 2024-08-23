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


import torch
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo_aligner.models.nlp.gpt.megatron_gpt_reward_model import MegatronGPTRewardModel
from nemo_aligner.utils.train_utils import set_sync_funcs


class MegatronGPTRegressionRankingRewardModel(MegatronGPTRewardModel):
    """
    Megatron GPT Regression Reward Model Training. 
    Regression reward models use a MSE loss to fit multi-attribute numeric labels for each data point.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.attribute_weights = torch.Tensor(self.cfg.regression.attribute_weights).unsqueeze(-1)

        if self.enable_standardization:
            raise NotImplementedError("Reward Standardization is not supported for regression reward models")

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(dataloader_iter, model):

            batch = next(dataloader_iter)
            is_binary = batch["is_binary"].cuda()

            required_keys = set()
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                # there is a problem with apex ignoring the mask on the older models
                # so we will always give the attention mask
                required_keys.add("attention_mask")

                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(("inputs", "position_ids", "chosen", "rejected"))

                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(("labels", "lengths", "loss_mask", "chosen_length", "rejected_length"))

            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}
            
            ###################################
            #     Regression Forward
            ###################################

            regression_forward_args = {
                "input_ids": batch["inputs"],
                "lengths": batch["lengths"],
                "position_ids": batch["position_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": None,
            }

            output_tensor = model(**regression_forward_args)
            # in this nemo version the model and autocast dtypes are not synced
            # so we need to explicitly cast it
            if not parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor.to(dtype=self.autocast_dtype)
            

            ###################################
            #     Ranking Forward
            ###################################

            # only do the torch.cat if it's available
            lengths, tokens = None, None
            position_ids = (
                torch.cat((batch["position_ids"], batch["position_ids"]), dim=0)
                if batch["position_ids"] is not None
                else None
            )

            if batch["chosen_length"] is not None and batch["rejected_length"] is not None:
                # Combine chosen and rejected lengths and then tokens.
                lengths = torch.cat((batch["chosen_length"], batch["rejected_length"]), dim=0)

            if batch["chosen"] is not None and batch["rejected"] is not None:
                tokens = torch.cat((batch["chosen"], batch["rejected"]), dim=0)

            ranking_forward_args = {
                "input_ids": tokens,
                "lengths": lengths,
                "position_ids": position_ids,
                "attention_mask": batch["attention_mask"],
                "labels": None,
            }

            ranking_output_tensor = model(**ranking_forward_args) @ self.attribute_weights.cuda()

            # in this nemo version the model and autocast dtypes are not synced
            # so we need to explicitly cast it
            if not parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor.to(dtype=self.autocast_dtype)

            @torch.no_grad()
            def gather_and_split_rewards(rewards_out):
                rewards_out = rewards_out.detach()

                dp_group = parallel_state.get_data_parallel_group()
                output_list = [torch.zeros_like(rewards_out) for _ in range(dp_group.size())]

                # gather it to compute the std later on
                torch.distributed.all_gather(output_list, output_tensor, group=dp_group)

                # output_list is a list of tensors with len == number of DP workers
                # we need to split each of these tensors and concat them back into a single tensor for chosen and rejected rewards
                split_iter = map(self.split_output_tensor, output_list)

                # go from [(out_chosen_dp0, out_rejected_dp0), (out_chosen_dp1, out_rejected_dp1)] ->
                # [out_chosen_dp0, out_chosen_dp1], [out_rejected_dp0, out_rejected_dp1]
                out_chosen, out_rejected = map(torch.cat, zip(*split_iter))

                return out_chosen.flatten(), out_rejected.flatten()

            def loss_func(output_tensor):
                # Regression Loss per micro batch (ub).
                regression_loss_for_ub = self.regression_loss_func(output_tensor, batch["labels"])

                # Ranking loss
                ranking_loss_for_ub, acc_chosen = self.ranking_loss_func(ranking_output_tensor)

                loss_for_ub = ((1 - is_binary) * regression_loss_for_ub.sum(-1)).sum() / self.cfg.regression.num_attributes / (1 - is_binary).sum()  + (is_binary * ranking_loss_for_ub).sum() / is_binary.sum()


                if validation_step and not self.cfg.data.get("validation_drop_last", True):
                    num_valid_tokens_in_ub = batch["loss_mask"].sum()
                    if loss_for_ub.isnan():
                        assert batch["loss_mask"].count_nonzero() == 0, "Got NaN loss with non-empty input"
                        loss_sum_for_ub = torch.zeros_like(num_valid_tokens_in_ub)
                    else:
                        loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

                    loss_sum_and_ub_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_ub.clone().detach().view(1),
                            torch.tensor([num_valid_tokens_in_ub]).cuda().clone().detach(),
                        ]
                    )
                    torch.distributed.all_reduce(
                        loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group()
                    )

                    return (
                        loss_for_ub,
                        {"loss_sum_and_ub_size": loss_sum_and_ub_size_all_gpu,},
                    )
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])

                    return (
                        loss_for_ub,
                        {"avg": reduced_loss,},
                    )

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def split_output_tensor(self, output_tensor):
        out_chosen, out_rejected = torch.split(output_tensor.float(), output_tensor.shape[0] // 2, dim=0)
        return out_chosen, out_rejected
    
    def ranking_loss_func(self, output_tensor):
        out_chosen, out_rejected = self.split_output_tensor(output_tensor)
        comp = out_chosen > out_rejected
        acc_chosen = torch.sum(comp) / comp.shape[0]
        # loss = -torch.nn.functional.logsigmoid(out_chosen - out_rejected).mean()
        loss = -torch.nn.functional.logsigmoid(out_chosen - out_rejected)
        return loss, acc_chosen

    def regression_loss_func(self, output_tensor, label_tensor):
        mask_val = self.cfg.get("loss_mask_val", -100.0)
        mask = label_tensor != mask_val
        num_valid_attributes = mask.float().sum()
        assert num_valid_attributes > 0, "Invalid sample: all attributes in label are masked, please check your data!"
        # Calculate the squared difference between prediction and label, and use the mask to ignore specific losses
        squared_diff = (output_tensor - label_tensor) ** 2 * mask
        # Calculate the mean of the masked squared differences
        # loss = squared_diff.sum() / num_valid_attributes
        return squared_diff

    def get_loss_and_metrics(self, batch, forward_only):
        data_iter = get_iterator_k_split(batch, get_num_microbatches())
        set_sync_funcs(self, forward_only)

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(forward_only),
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=self.cfg.encoder_seq_length,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # NOTE: assume that the returned values are already gathered across the DP workers
            # average loss across micro batches
            loss_tensors_list = [loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()

        else:
            loss_mean = torch.tensor(0.0, device=torch.cuda.current_device())

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(loss_mean, get_last_rank())

        metrics = {
            "loss": loss_mean,
        }

        # move to CPU
        metrics = {k: v.item() for k, v in metrics.items()}

        return loss_mean.item(), metrics
