from tgchatbot.audio.langid_ecapa import LangidEcapa
import pytest


@pytest.mark.parametrize("lang", ["en", "fr", "de", "ru"])
def test_langid_ecapa(lang, audio_data_dict):
    sample_rate = 16000
    net = LangidEcapa()
    audio_data = audio_data_dict[lang]
    pred_lang = net(audio_data, sample_rate)
    assert (type(pred_lang) == str)
    assert (len(pred_lang) > 0)
    print("Lang: {}->{}".format(lang, pred_lang))
