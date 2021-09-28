"""
    TTS from TensorSpeech.
"""

__all__ = ['TtsTensorspeech']


class TtsTensorspeech(object):
    """
    TTS from TensorSpeech.

    Parameters:
    ----------
    lang : str
        Language.
    tts_name : str, default 'tacotron2'
        TTS model name.
    use_cuda : bool, default False
        Whether to use CUDA.
    """
    def __init__(self,
                 lang,
                 tts_name="tacotron2",
                 use_cuda=False):
        super(TtsTensorspeech, self).__init__()
        # assert (lang in ("en", "fr", "de"))
        assert (lang in ("en", "fr"))
        assert (tts_name in ("tacotron2", "fastspeech2"))
        self.tts_name = tts_name

        if lang == "en":
            ds_lang = "ljspeech-en"
        elif lang == "fr":
            ds_lang = "synpaflex-fr"
        elif lang == "de":
            ds_lang = "thorsten-ger"
        else:
            raise ValueError("Unsupported language: {}".format(lang))

        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            if use_cuda:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                try:
                    tf.config.set_visible_devices([], "GPU")
                except RuntimeError:
                    pass

        from tensorflow_tts.inference import AutoProcessor
        from tensorflow_tts.inference import TFAutoModel

        self.processor = AutoProcessor.from_pretrained("tensorspeech/tts-{}-{}".format(tts_name, ds_lang))
        self.tts = TFAutoModel.from_pretrained("tensorspeech/tts-{}-{}".format(tts_name, ds_lang))

        vocoder_name = "mb_melgan"
        self.vocoder = TFAutoModel.from_pretrained("tensorspeech/tts-{}-{}".format(vocoder_name, ds_lang))

    def __call__(self, text):
        """
        Process an utterance.

        Parameters:
        ----------
        text : str
            Source utterance.

        Returns:
        -------
        np.array
            Destination audio.
        """
        input_tokens = self.processor.text_to_sequence(text)

        import tensorflow as tf

        if self.tts_name == "tacotron2":
            decoder_output, mel_outputs, stop_token_prediction, alignment_history = self.tts.inference(
                input_ids=tf.expand_dims(tf.convert_to_tensor(input_tokens, dtype=tf.int32), 0),
                input_lengths=tf.convert_to_tensor([len(input_tokens)], tf.int32),
                speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32))
        elif self.tts_name == "fastspeech2":
            mel_before, mel_outputs, duration_outputs, _, _ = self.tts.inference(
                input_ids=tf.expand_dims(tf.convert_to_tensor(input_tokens, dtype=tf.int32), 0),
                speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
                speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
                f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
                energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32))
        else:
            raise ValueError("Unsupported TTS model")

        audio = self.vocoder.inference(mel_outputs)[0, :, 0]
        return audio.numpy()
