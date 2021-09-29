FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04
LABEL maintainer="osemery@gmail.com"

RUN apt-get update && apt-get upgrade -y && apt-get autoremove
RUN apt-get install -y python3-pip

# For NeMo/librosa:
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && export DEBIAN_FRONTEND=noninteractive && apt-get install -y tzdata && dpkg-reconfigure --frontend noninteractive tzdata
RUN apt-get install -y libsndfile1-dev ffmpeg

# For TensorFlowTTS:
RUN apt-get install -y git

COPY . /root/projects/tgchatbot/
WORKDIR /root/projects/tgchatbot/

RUN pip install --upgrade pip setuptools wheel
RUN pip install torch==1.9.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt
RUN pip install TensorFlowTTS==1.8
RUN pip install huggingface-hub==0.0.17 six==1.16.0 numpy==1.20.3 llvmlite==0.37.0 numba==0.54.0 typing-extensions==3.10.0.2 h5py==3.4.0

RUN pip install pytest
RUN pip install .
RUN pytest
WORKDIR /root/projects/
RUN rm -rf /root/projects/tgchatbot/

ENTRYPOINT ["python3", "-m", "tgchatbot.launch"]
