ARG SOURCE=ghcr.io/neuralmagic/cuda-python3.10

ARG TORCH_VERSION=2.1.2
ARG TORCHVISION_VERSION=0.16.2
ARG CUDA=121
ARG VERSION

FROM $SOURCE

ARG VERSION

ARG CUDA
ARG TORCH_VERSION
ARG TORCHVISION_VERSION

RUN python3.10 -m pip install --upgrade pip \
    && python3.10 -m pip install --upgrade setuptools

RUN python3.10 -m pip install torch==${TORCH_VERSION}+cu${CUDA} torchvision==${TORCHVISION_VERSION}+cu${CUDA} -f https://download.pytorch.org/whl/torch_stable.html \
    && python3.10 -m pip install --no-cache-dir "sparseml[onnxruntime,torchvision,transformers,yolov5,ultralytics]==$VERSION" 

HEALTHCHECK CMD python3.10 -c 'import sparseml'
RUN python3.10 -m pip list | grep llmcompressor
CMD bash