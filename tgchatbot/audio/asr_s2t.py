"""
    ASR based on Speech to Text Transformer model.
"""

__all__ = ['AsrS2t']


class AsrS2t(object):
    """
    ASR based on Speech to Text Transformer model.

    Parameters:
    ----------
    src_lang : str, default 'en'
        Original language.
    dst_lang : str, default 'en'
        Target language.
    use_cuda : bool, default False
        Whether to use CUDA.
    """
    def __init__(self,
                 src_lang="en",
                 dst_lang="en",
                 use_cuda=False):
        super(AsrS2t, self).__init__()
        assert (src_lang in ("en", "fr", "de"))
        assert (dst_lang in ("en", "fr", "de", "ru"))
        self.src_lang = src_lang
        self.dst_lang = dst_lang
        self.use_cuda = use_cuda

        if src_lang == "en":
            if dst_lang != "en":
                asr_model_name = "facebook/s2t-medium-mustc-multilingual-st"
            else:
                asr_model_name = "facebook/s2t-medium-librispeech-asr"
            self.sampling_rate = 16_000
        elif (src_lang in ("fr", "de")) and (dst_lang == "en"):
            asr_model_name = "facebook/s2t-small-covost2-{}-en-st".format(src_lang)
            self.sampling_rate = 48_000
        else:
            ValueError("Unsupported languages: from `{}` to `{}`".format(src_lang, dst_lang))

        from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
        self.processor = Speech2TextProcessor.from_pretrained(asr_model_name)
        self.model = Speech2TextForConditionalGeneration.from_pretrained(asr_model_name)

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
            audio_features = self.processor(input_audio, sampling_rate=self.sampling_rate, return_tensors="pt")
            input_values = audio_features["input_features"]
            attention_mask = audio_features["attention_mask"]
            if self.use_cuda:
                input_values = input_values.cuda()
                attention_mask = attention_mask.cuda()
            forced_bos_token_id = self.processor.tokenizer.lang_code_to_id[self.dst_lang]\
                if self.dst_lang != "en" else None
            predicted_tokens = self.model.generate(
                input_ids=input_values,
                attention_mask=attention_mask,
                forced_bos_token_id=forced_bos_token_id)
            predicted_texts = self.processor.batch_decode(predicted_tokens, skip_special_tokens=True)

        text = predicted_texts[0]
        if self.dst_lang != "en":
            start = len(text.split(":")[0]) + 2
            text = text[start:]
        return text
