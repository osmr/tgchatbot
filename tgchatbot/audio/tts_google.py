"""
    TTS from Google Cloud.
"""

__all__ = ['TtsGoogle']

import numpy as np
from google.cloud import texttospeech


class TtsGoogle(object):
    """
    TTS from Google Cloud.
    See https://cloud.google.com/text-to-speech/docs.

    Parameters:
    ----------
    lang : str
        Language.
    use_cuda : bool, default False
        Whether to use CUDA (fake argument for a Cloud Service).
    """
    def __init__(self,
                 lang,
                 use_cuda=False):
        super(TtsGoogle, self).__init__()
        assert (lang in ("en", "ru"))
        self.lang = lang
        self.use_cuda = use_cuda

        lang_code_dict = {"en": "en-US", "ru": "ru-RU"}
        self.lang_code = lang_code_dict[lang]

        self.client = texttospeech.TextToSpeechClient()

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
        response = self.client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=texttospeech.VoiceSelectionParams(
                language_code=self.lang_code,
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sampleRateHertz=22050))
        if response.status_code != 200:
            raise RuntimeError("Invalid response received: code: {}, message: {}".format(
                response.status_code, response.text))
        audio_array = np.frombuffer(response.content.decode("UTF-8"))
        return audio_array
