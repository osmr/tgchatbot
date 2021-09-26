# Telegram chatbot 
This is a chatbot for the [Telegram](https://telegram.org) instance messenger, which uses various free conversational
Natural Language Understanding, Automatic Speech Recognition and Text-To-Speech neural networks for different languages.

## Used neural networks

### Text Conversational networks
1. Facebook's BlenderBot ([English](https://huggingface.co/facebook/blenderbot-400M-distill))
2. Microsoft's DialogGPT ([English](https://github.com/microsoft/DialoGPT), [French](https://huggingface.co/cedpsam/chatbot_fr), [Russian](https://huggingface.co/Grossmend/rudialogpt3_medium_based_on_gpt2))

### Text Neural Machine Translation
- MarianMT/OpusMT [models](https://github.com/Helsinki-NLP/Opus-MT)

### Text Punctuation and Capitalization
- NVIDIA NeMo Bert [models](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html)

### Automatic Speech Recognition
1. PytorchCV QuartzNet [models](https://github.com/osmr/imgclsmob)
2. Facebook's Wav2Vec2 XLSR-53 [models](https://github.com/jonatasgrosman/wav2vec2-sprint)
3. Facebook's Speech to Text Transformer [models](https://github.com/pytorch/fairseq/tree/main/examples/speech_to_text)

### Text-To-Speech
1. NVIDIA NeMo TTS [models](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/tts/intro.html)
2. TensorSpeech TensorFlowTTS [models](https://github.com/TensorSpeech/TensorFlowTTS)

### Audio Language Identification
- ECAPA-TDNN [model](https://huggingface.co/TalTechNLP/voxlingua107-epaca-tdnn-ce)

### Audio Emotion Recognition
1. S3PRL's Wav2Vec2 based [model](https://huggingface.co/superb/wav2vec2-base-superb-er)
2. S3PRL's Hubert based [model](https://huggingface.co/superb/hubert-large-superb-er)

## Installation


