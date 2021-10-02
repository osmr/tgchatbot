"""
    TTS from Amazon Cloud (Polly).
"""

__all__ = ['TtsAmazon']

import numpy as np


class TtsAmazon(object):
    """
    TTS from Amazon Cloud (Polly).
    See https://docs.aws.amazon.com/polly/latest/dg/what-is.html.

    Parameters:
    ----------
    lang : str
        Language.
    aws_access_key_id : str
        AWS access key ID.
    aws_secret_access_key : str
        AWS secret access key.
    aws_region_name : str
        AWS region when creating new connections.
    use_cuda : bool, default False
        Whether to use CUDA (fake argument for a Cloud Service).
    """
    def __init__(self,
                 lang,
                 aws_access_key_id,
                 aws_secret_access_key,
                 aws_region_name,
                 use_cuda=False):
        super(TtsAmazon, self).__init__()
        assert (lang in ("en", "fr", "de", "ru"))
        self.lang = lang
        self.use_cuda = use_cuda

        lang_code_dict = {"en": "en-US", "fr": "fr-FR", "de": "de-DE", "ru": "ru-RU"}
        voice_id_dict = {"en": "Joanna", "fr": "Léa", "de": "Vicki", "ru": "Tatyana"}

        self.lang_code = lang_code_dict[lang]
        self.voice_id = voice_id_dict[lang]

        from boto3 import Session
        session = Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region_name)
        self.client = session.client(service_name="polly")

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
        from botocore.exceptions import BotoCoreError, ClientError
        from contextlib import closing

        try:
            response = self.client.synthesize_speech(
                Engine="standard",
                LanguageCode=self.lang_code,
                OutputFormat="pcm",
                SampleRate="22050",
                Text=text,
                VoiceId=self.voice_id)
        except (BotoCoreError, ClientError) as er:
            raise RuntimeError("The service returned an error: {}".format(er))

        if "AudioStream" not in response:
            raise RuntimeError("The response didn't contain audio data")

        with closing(response["AudioStream"]) as stream:
            audio_array = np.frombuffer(stream.read())

        return audio_array
