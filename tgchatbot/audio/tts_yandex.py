"""
    TTS from Yandex.
"""

__all__ = ['TtsYandex']

import requests
import numpy as np
from io import BytesIO
from librosa.core import resample as lr_resample
from .audio_yandex import AudioYandex
from .audio_converter import AudioConverter


class TtsYandex(AudioYandex):
    """
    TTS from Yandex.
    """
    def __init__(self, **kwargs):
        super(TtsYandex, self).__init__(**kwargs)

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
        sample_rate = 48000
        response = requests.post(
            url=AudioYandex.TTS_URL,
            headers={"Authorization": "Bearer {}".format(self.iam_token)},
            data={
                "text": text,
                "lang": self.lang_code,
                # "voice": "oksana",
                # "emotion": "neutral",
                # "speed": 1.0,
                "format": self.audio_format,
                "sampleRateHertz": sample_rate,
                "folderId": self.folder_id},
            stream=True)
        if response.status_code != 200:
            raise RuntimeError("Invalid response received: code: {}, message: {}".format(
                response.status_code, response.text))
        if self.use_ogg_audio_format:
            with BytesIO() as b:
                b.write(response.content)
                b.seek(0)
                audio_array = AudioConverter.read_from_buffer(b)
        else:
            audio_array = np.frombuffer(response.content)
            audio_array = lr_resample(y=audio_array, orig_sr=sample_rate, target_sr=22050)

        return audio_array
