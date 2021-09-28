from tgchatbot.audio.tts_tensorspeech import TtsTensorspeech
from tgchatbot.audio.asr_quartznet import AsrQuartznet
import soundfile as sf
import librosa
import pytest


@pytest.fixture(scope="module")
def utterance_dict():
    utterance_dict = {
        "en": "Recent research at Harvard has shown meditating for as little as 8 weeks, can actually increase the "
              "grey matter in the parts of the brain responsible for emotional regulation, and learning.",
        "fr": "Oh, je voudrais tant que tu te souviennes Des jours heureux quand nous étions amis",
        # "de": "Möchtest du das meiner Frau erklären? Nein? Ich auch nicht.",
    }
    return utterance_dict


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_tts_tensorspeech(lang, use_cuda, utterance_dict, tmp_path):
    text = utterance_dict[lang]
    tts_sample_rate = 22050
    model = TtsTensorspeech(lang=lang, use_cuda=use_cuda)
    audio_data = model(text)
    assert (len(audio_data) > 0)
    audio_file_path = tmp_path / "audio_ttd_tensorspeech_{}.wav".format(lang)
    sf.write(
        file=audio_file_path,
        data=audio_data,
        samplerate=tts_sample_rate,
        subtype="PCM_16")

    asr_sample_rate = 16000
    asr_audio_data = librosa.load(path=audio_file_path, sr=asr_sample_rate, mono=True)[0]
    asr = AsrQuartznet(lang=lang, use_cuda=use_cuda)
    asr_text = asr(asr_audio_data)
    assert (type(asr_text) == str)
    assert (len(asr_text) > 0)
    assert (0.5 < float(len(asr_text)) / len(text) < 2.0)
    print("Src text: {}".format(text))
    print("Dst text: {}".format(asr_text))
