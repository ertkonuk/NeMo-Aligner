# Docker image.
# FROM nvcr.io/ea-bignlp/nemofw-training:23.07-py3
# FROM nvcr.io/nvidia/pytorch:23.06-py3 
# FROM gitlab-master.nvidia.com/dl/joc/nemo-rlhf:240923-2308updated-pyt-030
# FROM gitlab-master.nvidia.com/dl/joc/nemo-rlhf:270923-2308updated-pyt-030
# FROM gitlab-master.nvidia.com/akhattar/nemo_containers:value_ds
# FROM gitlab-master.nvidia.com/dl/joc/nemo-rlhf:102524-pyt040
# FROM gitlab-master.nvidia.com/dl/joc/nemo-rlhf:2308-021123-pyt040
# FROM gitlab-master.nvidia.com/dl/joc/nemo-rlhf:2308-stable-pyt040
# FROM gitlab-master.nvidia.com/dl/joc/nemo-rlhf:degert-test
# FROM nvcr.io/nvidian/nemo:public
# FROM gitlab-master.nvidia.com/dl/joc/nemo-ci/train:pipe.11040183
# FROM nvcr.io/nvidian/nemo:aligner
# FROM gitlab-master.nvidia.com/dl/joc/nemo-aligner:v2-022924-nemo-1.23.0
FROM gitlab-master.nvidia.com/dl/joc/nemo-ci/main/train:pipe.16440368-x86


# Testing
# FROM nvcr.io/ea-bignlp/ga-participants/nemofw-training:23.08.02
# FROM nvcr.io/ea-bignlp/beta-inf-prerelease/infer:23.10.v1
# FROM nvcr.io/ea-bignlp/ga-participants/nemofw-inference:23.10

# Official NeMo GA image
# FROM nvcr.io/ea-bignlp/ga-participants/nemofw-training:23.08.02
# FROM nvcr.io/ea-bignlp/ga-participants/nemofw-training:23.11

# User settings.
ARG USERNAME=abukharin
ARG USER_UID=1003
ARG USER_GID=1003

# Create the user.
RUN groupadd --gid $USER_GID $USERNAME
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# My custom setup.
RUN echo '~/src/perso/2020/"2020-01-16 - Config files"/docker_setup.sh' >> /home/abukharin/.bashrc

# Set default user.
USER $USERNAME