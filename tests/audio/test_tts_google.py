from tgchatbot.audio.tts_google import TtsGoogle
import os
import pytest


@pytest.mark.parametrize("lang", ["en", "ru"])
def test_tts_google(lang, pytestconfig):
    google_credentials = pytestconfig.getoption("google_credentials")

    if google_credentials is not None:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials
        test_dict = {
            "en": "Hello",
            "ru": "Привет",
        }
        text = test_dict[lang]
        model = TtsGoogle(lang=lang)
        audio_data = model(text)
        assert (len(audio_data) > 0)
