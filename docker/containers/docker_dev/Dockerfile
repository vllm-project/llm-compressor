ARG SOURCE=ghcr.io/neuralmagic/cuda-python3.10

ARG TORCH_VERSION=2.1.2
ARG TORCHVISION_VERSION=0.16.2
ARG CUDA=121
ARG BRANCH

FROM $SOURCE

ARG BRANCH

RUN python3.10 -m pip install --upgrade pip \
    && python3.10 -m pip install --upgrade setuptools

ARG CUDA
ARG TORCH_VERSION
ARG TORCHVISION_VERSION

RUN python3.10 -m pip install torch==${TORCH_VERSION}+cu${CUDA} torchvision==${TORCHVISION_VERSION}+cu${CUDA} -f https://download.pytorch.org/whl/torch_stable.html \
    && git clone https://github.com/neuralmagic/sparseml.git --depth 1 --single-branch -b ${BRANCH} \
    && python3.10 -m pip install -e "./sparseml[dev]"

HEALTHCHECK CMD python3.10 -c 'import sparseml'
RUN python3.10 -m pip list | grep llmcompressor
CMD bash