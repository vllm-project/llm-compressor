ARG TORCH_VERSION=1.9.1
ARG TORCHVISION_VERSION=0.10.1
ARG CUDA_VERSION=11.1.1
ARG VERSION
ARG DEPS="all"
ARG VENV="/venv"
ARG BRANCH

# Setup the base image & install dependencies
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu18.04 as base
# As of 05/05/22 nvidia images are broken. Two lines below are a temporary fix.
# Source: https://github.com/NVIDIA/nvidia-docker/issues/1632

ARG VENV
RUN if [ -f /etc/apt/sources.list.d/cuda.list ] ; then rm /etc/apt/sources.list.d/cuda.list ; fi
RUN if [ -f /etc/apt/sources.list.d/nvidia-ml.list ] ; then rm /etc/apt/sources.list.d/nvidia-ml.list ; fi

RUN set -Eeuxo \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
        git-all \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8 python3.8-venv python3-pip\
        build-essential libssl-dev libffi-dev \
        ffmpeg libsm6 libxext6 \
        curl 

# Install virtualenv and create a virtual environment
RUN set -Eeuxo \
    && apt-get update \
    && python3.8 -m pip install --upgrade virtualenv \
    && python3.8 -m virtualenv --python=/usr/bin/python3.8 $VENV \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add the virtual environment to the $PATH
ENV PATH="${VENV}/bin:$PATH"

FROM base as cuda-10.2
ARG TORCH_VERSION
ARG VENV
ARG TORCHVISION_VERSION
RUN $VENV/bin/pip install --upgrade pip torch==${TORCH_VERSION}+cu102 torchvision==${TORCHVISION_VERSION}+cu102  -f https://download.pytorch.org/whl/torch_stable.html

FROM base as cuda-11.1.1
ARG TORCH_VERSION
ARG VENV
ARG TORCHVISION_VERSION
RUN $VENV/bin/pip install --upgrade pip torch==${TORCH_VERSION}+cu111 torchvision==${TORCHVISION_VERSION}+cu111 -f https://download.pytorch.org/whl/torch_stable.html

FROM cuda-$CUDA_VERSION as cuda_builder
ARG VENV
RUN $VENV/bin/pip install --upgrade setuptools wheel
ARG BRANCH
RUN \
   if [ -z "${BRANCH}"] ; then  \
    echo Will install from pypi based on mode and version ; \
  else  \
    echo cloning from "${BRANCH}" &&  \
    git clone https://github.com/neuralmagic/sparseml.git --depth 1 -b "${BRANCH}" ; \
  fi;


FROM cuda_builder AS container_branch_base
ARG VENV
ENV PATH="${VENV}/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=200
ARG VERSION
ARG MODE=""
ARG BRANCH

RUN \
    if [ -n "$BRANCH" ] ; then \
      echo Installing from BRANCH && \
      $VENV/bin/pip install --no-cache-dir "./sparseml"; \
    elif [ "$MODE" = "nightly" ] ; then \
      echo Installing nightlies ... && \
      if [ -z "$VERSION" ] ; then \
        echo Installing latest nightlies ...  && \
        $VENV/bin/pip install --no-cache-dir "sparseml-nightly"; \
      else \
        echo Installing nightlies ... with $VERSION && \
        $VENV/bin/pip install --no-cache-dir "sparseml-nightly==$VERSION"; \
      fi; \
    elif [ -z "$VERSION" ] ; then \
      echo Installing latest llmcompressor ... from pypi && \
      $VENV/bin/pip install --no-cache-dir "sparseml"; \
    else \
      echo Installing latest llmcompressor version="$VERSION" from pypi && \
      $VENV/bin/pip install --no-cache-dir "sparseml==$VERSION"; \
    fi;

FROM cuda_builder AS container_branch_all
ARG VENV
ENV PATH="${VENV}/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=200
ARG VERSION
ARG MODE
ARG BRANCH

RUN \
    if [ -n "$BRANCH" ] ; then \
      echo Installing from BRANCH && \
      $VENV/bin/pip install --no-cache-dir "./sparseml[onnxruntime,torchvision,transformers,yolov5,ultralytics]"; \
    elif [ "$MODE" = "nightly" ] ; then \
      if [ -z $VERSION] ; then \
        $VENV/bin/pip install --no-cache-dir "sparseml-nightly[onnxruntime,torchvision,transformers,yolov5,ultralytics]"; \
      else \
        $VENV/bin/pip install --no-cache-dir "sparseml-nightly[onnxruntime,torchvision,transformers,yolov5,ultralytics]==$VERSION"; \
      fi; \
    elif [ -z $VERSION] ; then \
      $VENV/bin/pip install --no-cache-dir "sparseml[onnxruntime,torchvision,transformers,yolov5,ultralytics]"; \
    else \
      $VENV/bin/pip install --no-cache-dir "sparseml[onnxruntime,torchvision,transformers,yolov5,ultralytics]==$VERSION"; \
    fi;


FROM cuda_builder AS container_branch_dev
ARG VENV
ENV PATH="${VENV}/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=200
ARG VERSION
ARG MODE
ARG BRANCH

RUN \
    if [ -n "$BRANCH" ] ; then \
      echo Installing from BRANCH with editable mode && \
      $VENV/bin/pip install -e  "./sparseml[dev]"; \
    else \
      echo Installing from main  with editable mode && \
      git clone https://github.com/neuralmagic/sparseml.git --depth 1 -b main && \
      $VENV/bin/pip install -e "./sparseml[dev]"; \
    fi;

FROM container_branch_${DEPS} AS build
RUN echo Build complete, going onto prod

FROM base as dev
ARG VENV
COPY --from=build $VENV $VENV
COPY --from=build sparseml sparseml
ENV PATH="${VENV}/bin:$PATH"
HEALTHCHECK CMD python -c 'import sparseml'
CMD bash

FROM base as prod
ARG VENV
COPY --from=build $VENV $VENV
ENV PATH="${VENV}/bin:$PATH"
HEALTHCHECK CMD python -c 'import sparseml'
RUN pip list | grep llmcompressor
CMD bash

