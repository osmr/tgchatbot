from tgchatbot.audio.tts_amazon import TtsAmazon
import pytest


@pytest.mark.parametrize("lang", ["en", "fr", "de", "ru"])
def test_tts_amazon(lang, pytestconfig):
    aws_access_key_id = pytestconfig.getoption("aws_access_key_id")
    aws_secret_access_key = pytestconfig.getoption("aws_secret_access_key")
    aws_region_name = pytestconfig.getoption("aws_region_name")

    if (aws_access_key_id is not None) and (aws_secret_access_key is not None) and (aws_region_name is not None):
        test_dict = {
            "en": "Hello. Do you speak a foreign language? One language is never enough.",
            "fr": "Bonjour. Parlez-vous une autre langue que le français? Une langue n'est jamais assez.",
            "de": "Hallo. Sprechen Sie eine Fremdsprache? Eine Sprache ist nie genug.",
            "ru": "Привет. Вы говорите на иностранном языке? Одного языка никогда не бывает достоточно.",
        }
        text = test_dict[lang]
        model = TtsAmazon(
            lang=lang,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region_name=aws_region_name)
        audio_data = model(text)
        assert (len(audio_data) > 0)
