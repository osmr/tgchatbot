from tgchatbot.audio.asr_amazon import AsrAmazon
import pytest


@pytest.mark.parametrize("lang", ["en", "fr", "de"])
def test_asr_amazon(lang, audio_data_dict, pytestconfig):
    aws_access_key_id = pytestconfig.getoption("aws_access_key_id")
    aws_secret_access_key = pytestconfig.getoption("aws_secret_access_key")
    aws_region_name = pytestconfig.getoption("aws_region_name")

    if (aws_access_key_id is not None) and (aws_secret_access_key is not None) and (aws_region_name is not None):
        asr = AsrAmazon(
            lang=lang,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region_name=aws_region_name)
        audio_data = audio_data_dict[lang]
        text = asr(audio_data)
        assert (type(text) == str)
        assert (len(text) > 0)
        print("Text: {}".format(text))
