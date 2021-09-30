"""
    Base class from Yandex ASR/TTS.
"""

__all__ = ['AudioYandex']

import json
import requests


class AudioYandex(object):
    TTS_URL = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
    LANG_CODE_DICT = {"en": "en-US", "ru": "ru-RU", "tr": "tr-TR"}

    """
    Base class from Yandex ASR/TTS.

    Parameters:
    ----------
    lang : str
        Language.
    oauth_token : str
        Yandex OAuth token.
    iam_token : str
        Yandex IAM token.
    folder_id : str
        Yandex Cloud folder id.
    use_ogg_audio_format : bool, default False
        Whether to use OGG audio format in communications.
    use_cuda : bool, default False
        Whether to use CUDA (fake argument for a Cloud Service).
    """
    def __init__(self,
                 lang,
                 oauth_token,
                 iam_token,
                 folder_id,
                 use_ogg_audio_format=False,
                 use_cuda=False):
        super(AudioYandex, self).__init__()
        assert (lang in ("en", "ru", "tr"))
        assert (folder_id is not None) and (folder_id != "")
        assert (use_cuda is not None)

        self.lang_code = AudioYandex.LANG_CODE_DICT[lang]
        self.folder_id = folder_id

        if (iam_token is None) or (iam_token == "") or\
                not AudioYandex.is_iam_token_valid(iam_token=iam_token, folder_id=folder_id):
            assert (oauth_token is not None) and (oauth_token != "")
            iam_token, _ = AudioYandex.create_iam_token(oauth_token)

        self.iam_token = iam_token

        self.use_ogg_audio_format = use_ogg_audio_format
        self.audio_format = "oggopus" if use_ogg_audio_format else "lpcm"

    @staticmethod
    def create_iam_token(oauth_token):
        """
        Create a new IAM token.

        Parameters:
        ----------
        oauth_token : str
            OAuth token.

        Returns:
        -------
        iam_token : str
            IAM token.
        expires_at : str
            Expiration date.
        """
        response = requests.post(
            url="https://iam.api.cloud.yandex.net/iam/v1/tokens",
            params={"yandexPassportOauthToken": oauth_token})
        response_value = response.content.decode("UTF-8")
        response_dict = json.loads(response_value)
        iam_token = response_dict.get("iamToken")
        expires_at = response_dict.get("expiresAt")
        return iam_token, expires_at

    @staticmethod
    def is_iam_token_valid(iam_token, folder_id):
        """
        Validation check for the IAM token.

        Parameters:
        ----------
        iam_token : str
            IAM token.
        folder_id : str
            Yandex Cloud folder id.

        Returns:
        -------
        bool
            Validity.
        """
        response = requests.post(
            url=AudioYandex.TTS_URL,
            headers={"Authorization": "Bearer {}".format(iam_token)},
            data={"text": "check", "lang": "en-US", "folderId": folder_id})
        return response.status_code == 200
