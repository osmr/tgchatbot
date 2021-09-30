from tgchatbot.audio.asr_yandex import AsrYandex
import pytest


@pytest.mark.parametrize("lang", ["en", "ru"])
def test_asr_yandex(lang, audio_data_dict, pytestconfig):
    oauth_token = pytestconfig.getoption("yandex_oauth_token")
    folder_id = pytestconfig.getoption("yandex_folder_id")
    iam_token = pytestconfig.getoption("yandex_iam_token")

    if ((oauth_token is not None) or (iam_token is not None)) and (folder_id is not None):
        asr = AsrYandex(
            lang=lang,
            oauth_token=oauth_token,
            iam_token=iam_token,
            folder_id=folder_id)
        audio_data = audio_data_dict[lang]
        text = asr(audio_data)
        assert (type(text) == str)
        assert (len(text) > 0)
        print("Text: {}".format(text))
