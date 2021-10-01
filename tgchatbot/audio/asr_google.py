"""
    ASR from Google Cloud.
"""

__all__ = ['AsrGoogle']

from google.cloud import speech


class AsrGoogle(object):
    """
    ASR from Google Cloud.
    See https://cloud.google.com/speech-to-text/docs.

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
        super(AsrGoogle, self).__init__()
        assert (lang in ("en", "ru"))
        self.lang = lang
        self.use_cuda = use_cuda

        lang_code_dict = {"en": "en-US", "ru": "ru-RU"}
        self.lang_code = lang_code_dict[lang]

        self.client = speech.SpeechClient()

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
        operation = self.client.long_running_recognize(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=self.lang_code),
            audio=input_audio)
        response = operation.result(timeout=90)

        transcript_list = []
        for result in response.results:
            transcript_list.append(result.alternatives[0].transcript)
            print(u"Transcript: {}".format(result.alternatives[0].transcript))
        text = "".join(transcript_list)
        return text
