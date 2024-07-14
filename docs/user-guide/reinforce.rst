.. include:: /content/nemo.rsts

.. _model-aligner-reinforce:

Model Alignment by REINFORCE
@@@@@@@@@@@@@@@@@@@@@@@

In this tutorial, we will go through the process of aligning a NeMo framework model using the reinforce algorithm. These can be models such as LLaMa2 or Mistral. Our scripts will work the same way.

RLHF is usually preceded by a Supervised Fine-Tuning (SFT). We should first follow the :ref:`Prerequisite guide <prerequisite>` and the :ref:`SFT guide <sft>`. After obtaining the SFT model, we will also need to train a reward model as in :ref:`PPO guide <ppo>`. We will use the REINFORCE sampling algorithm on the `Anthropic-HH-RLHF <https://huggingface.co/datasets/Anthropic/hh-rlhf>`__ dataset.

REINFORCE Training
############

After you have fine-tuned a GPT model using Supervised Fine-Tuning (SFT), and trained a reward model as explained in the preceding section, you can start align the policy using REINFORCE.

During REINFORCE training, we conceptually have 2 models interacting with each other:

#. The Policy Network: This is the model we are training, and it should start from an SFT model.
#. The Reward Model (RM): This model takes a prompt concatenated with a response as input, and outputs a single scalar value: the reward, which the rejection sampling algorithm will try to maximize.

In the most optimized configuration, Aligner will run the actor and initial policy within the same job and the reward model in a separate job. It will then use cpu offloading to load back the corresponding model when needed.
   
The next section discusses how to launch each of these two jobs.

Launching the Reward Model and Critic Server
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

To launch the server:

.. code-block:: bash 

   #!/bin/bash
   CHECKPOINT_NEMO_FILE="/path/to/trained_rm.nemo"
   GPFS="/path/to/nemo-aligner-repo"

   RESULTS_DIR="critic_results_dir"

   export PYTHONPATH="${NEMO_RLHF_DIR}:${PYTHONPATH}" \
   && export HYDRA_FULL_ERROR=1 \
   && python -u examples/nlp/gpt/serve_reward_model.py \
      --config-path=${CRITIC_CONFIG_PATH} \
      --config-name=${CRITIC_CONFIG_NAME} \
      trainer.num_nodes=1 \
      trainer.devices=8 \
      ++model.tensor_model_parallel_size=4 \
      rm_model_file=${RM_NEMO_FILE} \
      inference.port=${CRITIC_PORT}


The above example launches the reward model server on 8 gpus and 1 node. Please make sure to change ``trainer.devices``, ``trainer.num_nodes`` depending on your model size and scale. Aligner will work on any scale. Also make sure to tune the `trainer.rs.inference_micro_batch_size` argument, this sets how big of a batch the RS actor is allowed to send to the critic per DP rank.

Launching the Initial Policy and RS Actor Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The RS Actor training job contains the master controller that makes the HTTP calls to all servers when needed. To launch the RS Actor and Initial Policy server:

.. code-block:: bash 

   GPFS="/path/to/nemo-aligner-repo"
   TRAIN_DATA_PATH="/path/to/train_prompts.jsonl"
   VALID_DATA_PATH="/path/to/test_prompts.jsonl"

   PRETRAINED_ACTOR_NEMO_FILE="/path/to/sft_checkpoint.nemo"

   RESULTS_DIR="actor_results_dir"

   export PYTRITON_HOME=/pytriton_cache \
   && export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
   && python -u examples/nlp/gpt/train_gpt_reinforce_actor.py \
      --config-path=${CONF_DIR} \
      --config-name=${CONFIG_NAME} \
      "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
      pretrained_checkpoint.restore_from_path=\"${ACTOR_NEMO_FILE}\" \
      exp_manager.checkpoint_callback_params.save_top_k=1 \
      exp_manager.explicit_log_dir=\"${ACTOR_LOG_DIR}\" \
      trainer.ppo.initial_policy_kl_penalty=0.01 \
      trainer.ppo.max_epochs=1 \
      trainer.ppo.max_steps=313 \
      trainer.ppo.val_check_interval=4 \
      trainer.num_nodes=8 \
      trainer.devices=8 \
      ++model.tensor_model_parallel_size=4 \
      model.global_batch_size=${ACTOR_GBS} \
      model.micro_batch_size=1 \
      model.optim.lr=\"\\\$\{multiply:${ACTOR_LR},1.001\}\" \
      model.optim.sched.warmup_steps=0 \
      model.optim.sched.constant_steps=312 \
      model.optim.sched.min_lr=${ACTOR_LR} \
      model.optim.weight_decay=0.01 \
      model.ppo.num_rollout_samples=${NUM_ROLLOUTS} \
      model.ppo.rollout_micro_batch_size=4 \
      model.ppo.forward_micro_batch_size=4 \
      model.ppo.val_rollout_micro_batch_size=4 \
      model.data.data_impl=jsonl \
      remote_critic_rm.reward_model.ip=${host_critic} \
      remote_critic_rm.reward_model.port=${CRITIC_PORT} \
      model.ppo.num_rollout_per_prompt=4 \
      trainer.ppo.baseline=\"RLOO\"

The above launches the initial and actor server on 1 node with 8 GPUs

Launching Both Servers for RLHF training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

You can use slurm to launch the 2 jobs and get them to coordinate together in a full RLHF job via the following:

.. code-block:: bash 

   #!/bin/bash
   #SBATCH -N 1 --ntasks-per-node 8 -A <<ACCOUNT>> -p <<PARTITION>> --job-name <<JOBNAME>> -t 4:00:00 --exclusive
   #SBATCH hetjob
   #SBATCH -N 1 --ntasks-per-node 8 -A <<ACCOUNT>> -p <<PARTITION>> --job-name <<JOBNAME>> -t 4:00:00 --exclusive

   NAME="2p_rs"

   # PARAMETERS
   RM_NEMO_FILE="/path/to/trained_rm.nemo"

   ACTOR_NEMO_FILE="/path/to/sft_model.nemo"

   TRAIN_DATA_PATH="/path/to/train_prompts.jsonl"
   VALID_DATA_PATH="/path/to/test_prompts.jsonl"

   RESULTS_DIR="/path/to/results_dir"
   mkdir -p $RESULTS_DIR

   GPFS="/path/to/nemo-aligner-repo"
   MOUNTS="--container-mounts=MOUNTS" # mounts

   CONTAINER=<<<CONTAINER>>> # use the latest NeMo Training container, Aligner will work there

   PROJECT=rs_run

   CRITIC_LOG_DIR="${RESULTS_DIR}/critic_results"
   CRITIC_OUTFILE="${CRITIC_LOG_DIR}/critic_output_%j_%t.log"
   CRITIC_ERRFILE="${CRITIC_LOG_DIR}/critic_error_%j_%t.err"
   CRITIC_PORT=5567
   CRITIC_CONFIG_PATH="${NEMO_RLHF_DIR}/examples/nlp/gpt/conf"
   CRITIC_CONFIG_NAME="inference_rm"

   CONF_DIR="${NEMO_RLHF_DIR}/examples/nlp/gpt/conf"
   CONFIG_NAME="gpt_rs_actor"

   mkdir -p $CRITIC_LOG_DIR

   CRITIC_NAME="${NAME}_critic"

   read -r -d '' cmd_critic_inference <<EOF
   cd ${GPFS} \
   && export PYTHONPATH="${NEMO_RLHF_DIR}:${PYTHONPATH}" \
   && export HYDRA_FULL_ERROR=1 \
   && python -u examples/nlp/gpt/serve_reward_model.py \
      --config-path=${CRITIC_CONFIG_PATH} \
      --config-name=${CRITIC_CONFIG_NAME} \
      trainer.num_nodes=1 \
      trainer.devices=8 \
      ++model.tensor_model_parallel_size=4 \
      rm_model_file=${RM_NEMO_FILE} \
      inference.port=${CRITIC_PORT}
   EOF

   srun --het-group=0 -o $CRITIC_OUTFILE -e $CRITIC_ERRFILE --container-image=${CONTAINER} $MOUNTS bash -c "${cmd_critic_inference}" &

   sleep 30

   ACTOR_LOG_DIR="${RESULTS_DIR}/actor_results"
   CHECKPOINT_DIR="${ACTOR_LOG_DIR}/checkpoints"
   TENSOBOARD_DIR="${ACTOR_LOG_DIR}/tensorboard"

   NUM_ROLLOUTS=16
   NORMALIZE="True"
   ACTOR_LR="1e-6"
   ACTOR_GBS=16

   mkdir -p $ACTOR_LOG_DIR
   mkdir -p $TENSOBOARD_DIR
   mkdir -p $CHECKPOINT_DIR

   ACTOR_NAME="${NAME}_actor"

   host_critic="$(scontrol show hostnames=$SLURM_JOB_NODELIST_HET_GROUP_0 | head -n1)"

   read -r -d '' cmd_reinforce <<EOF
   cd ${GPFS} \
   && export PYTRITON_HOME=/pytriton_cache \
   && export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
   && python -u examples/nlp/gpt/train_gpt_reinforce_actor.py \
      --config-path=${CONF_DIR} \
      --config-name=${CONFIG_NAME} \
      "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
      pretrained_checkpoint.restore_from_path=\"${ACTOR_NEMO_FILE}\" \
      exp_manager.checkpoint_callback_params.save_top_k=1 \
      exp_manager.explicit_log_dir=\"${ACTOR_LOG_DIR}\" \
      trainer.ppo.initial_policy_kl_penalty=0.01 \
      trainer.ppo.max_epochs=1 \
      trainer.ppo.max_steps=313 \
      trainer.ppo.val_check_interval=4 \
      trainer.num_nodes=8 \
      trainer.devices=8 \
      ++model.tensor_model_parallel_size=4 \
      model.global_batch_size=${ACTOR_GBS} \
      model.micro_batch_size=1 \
      model.optim.lr=\"\\\$\{multiply:${ACTOR_LR},1.001\}\" \
      model.optim.sched.warmup_steps=0 \
      model.optim.sched.constant_steps=312 \
      model.optim.sched.min_lr=${ACTOR_LR} \
      model.optim.weight_decay=0.01 \
      model.ppo.num_rollout_samples=${NUM_ROLLOUTS} \
      model.ppo.rollout_micro_batch_size=4 \
      model.ppo.forward_micro_batch_size=4 \
      model.ppo.val_rollout_micro_batch_size=4 \
      model.data.data_impl=jsonl \
      remote_critic_rm.reward_model.ip=${host_critic} \
      remote_critic_rm.reward_model.port=${CRITIC_PORT} \
      model.ppo.num_rollout_per_prompt=4 \
      trainer.ppo.baseline=\"RLOO\"
   EOF

   srun --het-group=1 -o $PPO_OUTFILE -e $PPO_ERRFILE --container-image=${CONTAINER} $MOUNTS bash -c "${cmd_reinforce}" &

   wait

The above script runs the reward model critic server on 1 node and the actor on 1 node.

It is important to launch all jobs with ``&`` after the srun command, to ensure they do not block each other. 

.. note::
   Make sure to change the critic arg ``trainer.rs.inference_micro_batch_size`` such that ``trainer.rs.inference_micro_batch_size * DP size <= model.rs.rollout_micro_batch_size``.

REINFORCE Variants
%%%%%%%%%%%
In the above examples, we set the baseline to RLOO, i.e. the leave one out baseline. Other options include "ReMax" and "none", however we find that RLOO works best.

REINFORCE Results
%%%%%%%%%%%

Once you've completed rejection sampling training, you can serve your model using the `megatron_gpt_eval.py <https://github.com/NVIDIA/NeMo/blob/8cd5f1c8e7d4fed9f4f946028cd02047c5d2296f/examples/nlp/language_modeling/megatron_gpt_eval.py#L4>`__ script from the NeMo codebase to run more rigorous evaluation of your trained model.
   
Scaling the tutorial to bigger models
#####################################

The above tutorial is a way to get started with RLHF but is not the most optimal performant or convergence configuration. 