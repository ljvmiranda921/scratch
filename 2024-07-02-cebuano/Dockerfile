FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE

WORKDIR /stage

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git
COPY requirements.txt /stage
RUN pip install flash-attn --no-build-isolation
RUN pip install -r requirements.txt

# Copy all files
COPY . /stage
