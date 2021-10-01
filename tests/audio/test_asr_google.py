from tgchatbot.audio.asr_google import AsrGoogle
import os
import pytest


@pytest.mark.parametrize("lang", ["en", "ru"])
def test_asr_google(lang, audio_data_dict, pytestconfig):
    google_credentials = pytestconfig.getoption("google_credentials")

    if google_credentials is not None:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials
        asr = AsrGoogle(lang=lang)
        audio_data = audio_data_dict[lang]
        text = asr(audio_data)
        assert (type(text) == str)
        assert (len(text) > 0)
        print("Text: {}".format(text))
