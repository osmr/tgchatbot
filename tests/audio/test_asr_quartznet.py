from tgchatbot.audio.asr_quartznet import AsrQuartznet
import pytest


@pytest.mark.parametrize("lang", ["en", "fr", "de", "ru"])
def test_asr_quartznet(lang, use_cuda, audio_data_dict):
    asr = AsrQuartznet(lang=lang, use_cuda=use_cuda)
    audio_data = audio_data_dict[lang]
    text = asr(audio_data)
    assert (type(text) == str)
    assert (len(text) > 0)
    print("Text: {}".format(text))
