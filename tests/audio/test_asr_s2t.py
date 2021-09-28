from tgchatbot.audio.asr_s2t import AsrS2t
import pytest


@pytest.mark.parametrize("lang", ["en", "fr", "de", "ru"])
def test_asr_s2t_from_en(lang, use_cuda, audio_data_dict):
    asr = AsrS2t(src_lang="en", dst_lang=lang, use_cuda=use_cuda)
    audio_data = audio_data_dict["en"]
    text = asr(audio_data)
    assert (type(text) == str)
    assert (len(text) > 0)
    print("Text: {}".format(text))


@pytest.mark.parametrize("lang", ["fr", "de"])
def test_asr_s2t_to_en(lang, use_cuda, audio_data_dict):
    asr = AsrS2t(src_lang=lang, dst_lang="en", use_cuda=use_cuda)
    audio_data = audio_data_dict[lang]
    text = asr(audio_data)
    assert (type(text) == str)
    assert (len(text) > 0)
    print("Text: {}".format(text))
