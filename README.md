# Telegram AI chatbot
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

```
