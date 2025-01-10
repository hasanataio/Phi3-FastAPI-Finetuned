FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo PATH="/root/miniconda3/bin":$PATH >> .bashrc \
    && exec bash \
    && conda --version

RUN conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
RUN conda activate unsloth_env

RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install --no-deps trl peft accelerate bitsandbytes

RUN pip install -r requirements.txt

CMD ["uvicorn", "phi_api:app", "--host", "0.0.0.0", "--port", "8006"]
