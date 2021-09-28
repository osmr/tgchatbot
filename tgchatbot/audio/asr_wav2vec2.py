"""
    ASR based on Wav2Vec2 XLSR-53 model.
"""

__all__ = ['AsrWav2vec2']


class AsrWav2vec2(object):
    """
    ASR based on Wav2Vec2 XLSR-53 model.

    Parameters:
    ----------
    lang : str
        Language.
    use_cuda : bool, default False
        Whether to use CUDA.
    """
    def __init__(self,
                 lang,
                 use_cuda=False):
        super(AsrWav2vec2, self).__init__()
        assert (lang in ("en", "fr", "de", "ru"))
        self.lang = lang
        self.use_cuda = use_cuda

        lang_names = {"en": "english", "fr": "french", "de": "german", "ru": "russian"}
        lang_name = lang_names[lang] if lang in lang_names else ValueError("Unsupported language: {}".format(lang))

        asr_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-{}".format(lang_name)

        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        self.processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(asr_model_name)

        if use_cuda:
            self.model = self.model.cuda()

    def __call__(self, input_audio):
        """
        Process an utterance.

        Parameters:
        ----------
        input_audio : np.array
            Source audio.

        Returns:
        -------
        str
            Destination text.
        """
        import torch
        with torch.no_grad():
            audio_features = self.processor(input_audio, sampling_rate=16_000, return_tensors="pt", padding=True)
            input_values = audio_features.input_values
            attention_mask = audio_features.attention_mask
            if self.use_cuda:
                input_values = input_values.cuda()
                attention_mask = attention_mask.cuda()
            logits = self.model(input_values=input_values, attention_mask=attention_mask).logits
            predicted_tokens = torch.argmax(logits, dim=-1)
            predicted_texts = self.processor.batch_decode(predicted_tokens)

        text = predicted_texts[0]
        return text
