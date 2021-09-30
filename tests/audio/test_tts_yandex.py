from tgchatbot.audio.tts_yandex import TtsYandex
import pytest


@pytest.mark.parametrize("lang", ["en", "ru", "tr"])
def test_tts_yandex(lang, pytestconfig):
    oauth_token = pytestconfig.getoption("yandex_oauth_token")
    folder_id = pytestconfig.getoption("yandex_folder_id")
    iam_token = pytestconfig.getoption("yandex_iam_token")

    if ((oauth_token is not None) or (iam_token is not None)) and (folder_id is not None):
        test_dict = {
            "en": "Hello",
            "ru": "Привет",
            "tr": "Merhaba",
        }
        text = test_dict[lang]
        model = TtsYandex(
            lang=lang,
            oauth_token=oauth_token,
            iam_token=iam_token,
            folder_id=folder_id)
        audio_data = model(text)
        assert (len(audio_data) > 0)
