"""
    ASR from Yandex (without GRPC).
"""

__all__ = ['AsrYandex']

import json
import requests
from .audio_yandex import AudioYandex
from .audio_converter import AudioConverter


class AsrYandex(AudioYandex):
    """
    ASR from Yandex (without GRPC).
    See https://cloud.yandex.com/en-ru/docs/speechkit/stt/.
    """
    def __init__(self, **kwargs):
        super(AsrYandex, self).__init__(**kwargs)

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
        sample_rate = 16000
        if self.use_ogg_audio_format:
            audio_data = AudioConverter.write_to_buffer(
                audio_array=input_audio,
                audio_sample_rate=sample_rate,
                format="ogg")
        else:
            audio_data = input_audio
        response = requests.post(
            url="https://stt.api.cloud.yandex.net/speech/v1/stt:recognize",
            headers={"Authorization": "Bearer {}".format(self.iam_token)},
            params={
                "lang": self.lang_code,
                # "topic": "general",
                "format": self.audio_format,
                "sampleRateHertz": sample_rate,
                "folderId": self.folder_id},
            data=audio_data)
        if response.status_code != 200:
            raise RuntimeError("Invalid response received: code: {}, message: {}".format(
                response.status_code, response.text))
        response_value = response.content.decode("UTF-8")
        response_dict = json.loads(response_value)
        error_code = response_dict.get("error_code")
        if error_code is not None:
            raise RuntimeError("Response contains an error code: {}".format(response_dict.get("error_code")))
        text = response_dict.get("result")
        return text
