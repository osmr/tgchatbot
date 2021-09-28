from tgchatbot.audio.emrec_superb import EmrecSuperb
import pytest


@pytest.mark.parametrize("lang", ["en", "fr", "de", "ru"])
@pytest.mark.parametrize("model_type", ["hubert", "wav2vec2"])
def test_emrec_superb(lang, model_type, audio_data_dict):
    net = EmrecSuperb()
    audio_data = audio_data_dict[lang]
    emotion = net(audio_data)
    assert (type(emotion) == str)
    assert (len(emotion) > 0)
    print("Emotion: {}".format(emotion))
