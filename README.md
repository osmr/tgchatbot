# Telegram AI chatbot

[![Build Status](https://travis-ci.org/osmr/imgclsmob.svg?branch=master)](https://travis-ci.org/osmr/imgclsmob)
[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%2C3.8-lightgrey.svg)](https://github.com/osmr/imgclsmob)

This is a chatbot for the [Telegram](https://telegram.org) instance messenger, which uses various free conversational
Natural Language Understanding, Automatic Speech Recognition and Text-To-Speech neural networks for different languages.

## Used neural networks
1. Text Conversational networks
   1. Facebook's BlenderBot ([English](https://huggingface.co/facebook/blenderbot-400M-distill))
   2. Microsoft's DialogGPT ([English](https://github.com/microsoft/DialoGPT), [French](https://huggingface.co/cedpsam/chatbot_fr), [Russian](https://huggingface.co/Grossmend/rudialogpt3_medium_based_on_gpt2))

2. Text Neural Machine Translation
   - MarianMT/OpusMT [models](https://github.com/Helsinki-NLP/Opus-MT)

3. Text Punctuation and Capitalization
   - NVIDIA NeMo Bert [models](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html)

4. Automatic Speech Recognition
   1. PytorchCV QuartzNet [models](https://github.com/osmr/imgclsmob)
   2. Facebook's Wav2Vec2 XLSR-53 [models](https://github.com/jonatasgrosman/wav2vec2-sprint)
   3. Facebook's Speech to Text Transformer [models](https://github.com/pytorch/fairseq/tree/main/examples/speech_to_text)

5. Text-To-Speech
   1. NVIDIA NeMo TTS [models](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/tts/intro.html)
   2. TensorSpeech TensorFlowTTS [models](https://github.com/TensorSpeech/TensorFlowTTS)

6. Audio Language Identification
   - ECAPA-TDNN [model](https://huggingface.co/TalTechNLP/voxlingua107-epaca-tdnn-ce)

7. Audio Emotion Recognition
   1. S3PRL's Wav2Vec2 based [model](https://huggingface.co/superb/wav2vec2-base-superb-er)
   2. S3PRL's Hubert based [model](https://huggingface.co/superb/hubert-large-superb-er)

## Deployment

### Docker way

1. Install docker engine (actual [instructions](https://docs.docker.com/engine/install/)):
```
sudo apt update
sudo apt upgrade -y
sudo apt dist-upgrade -y
sudo apt autoremove -y

sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt-get install docker-ce docker-ce-cli containerd.io

sudo systemctl status docker

sudo usermod -aG docker $USER
newgrp docker
```
2. Install NVIDIA Container Toolkit (actual [instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```
3. Build docker image:
```
git clone https://github.com/osmr/tgchatbot.git
cd tgchatbot
docker build -t tgchatbot .
```
4. Run docker container (fill the `token` value):
```
docker run -it --rm --name=tgchatbot1 tgchatbot --token="<Your token>"
```
or
```
docker run -it --rm --gpus=all --name=tgchatbot1 tgchatbot --token="<Your token>" --use-cuda
```

### Virtualenv way

NB: You need `Python` >= 3.7 due to requirements of the `aiogram` and `SpeechBrain` packages.

1. Install virtualenv (actual [instructions](https://virtualenv.pypa.io/en/latest/installation.html)):
```
sudo -H pip install --upgrade pip setuptools wheel
sudo -H pip install Cython
sudo -H pip install virtualenv
```
2. Clone repo, create and activate environment:
```
git clone git@github.com:osmr/tgchatbot.git
cd tgchatbot
virtualenv venv
source venv/bin/activate
```
3. Install dependencies:
```
pip install torch==1.9.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install TensorFlowTTS==1.8
pip install huggingface-hub==0.0.17 six==1.16.0
pip install --upgrade numpy llvmlite numba typing-extensions h5py
pip install pytest
pip install .
```
4. Run tests:
```
pytest
```
5. Run the chatbot (fill the `token` value):
```
python -m tgchatbot.launch --token="<YOU token>"
docker run -it --rm --name=tgchatbot1 tgchatbot --token="<Your token>"
```
or
```
python -m tgchatbot.launch --token="<Your token>" --use-cuda
```
6. Deactivate environment:
```
deactivate
```

## Chatbot commands in Telegram
1. `/start` or `/help` - Welcome information.
2. `/start en` - Set input/output language to `en` (English). It can be `en`, `fr`, `de`, and `ru`.
3. `/lang` - Show current input/output languages status.
4. `/lang_src` - Show current input (your messages or speech) language status.
5. `/lang_dst` - Show current output (chatbot's messages and speech) language status.
6. `/lang en` - Set input/output language to `en` (English).
7. `/lang_src en` - Set input language to `en` (English).
8. `/lang_dst en` - Set output language to `en` (English).
9. `/tts` - Show current Text-To-Speech (TTS) activity status. It can be `yes` or `no`.
10. `/tts yes` - Activate TTS.
11. `/context` - Show contest for text (NLU) chatbot context corresponding the current user and output language.
12. `/context ""` - Erase NLU chatbot contest.
13. `Hello` - Text `Hello` to the chatbot. You can text amy message. 
13. `<Audio with speech>` - Say something to the chatbot.
