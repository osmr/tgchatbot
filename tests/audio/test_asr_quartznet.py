from tgchatbot.audio.asr_quartznet import AsrQuartznet
import os.path
import librosa
import pytest


@pytest.fixture(scope="module")
def audio_data_dict():
    root_dir_path = "../../data"
    file_paths = {
        "en": "common_voice_en_1.mp3",
        "fr": "common_voice_fr_17299384.mp3",
        "de": "common_voice_de_17298952.mp3",
        "ru": "common_voice_ru_18849003.mp3",
    }
    sample_rate = 16000
    audio_data_dict = {lang: librosa.load(
        path=os.path.join(root_dir_path, file_paths[lang]), sr=sample_rate, mono=True)[0] for lang in file_paths}
    return audio_data_dict


@pytest.mark.parametrize("lang", ["en", "fr", "de", "ru"])
@pytest.mark.parametrize("use_cuda", [False, True])
def test_asr_quartznet(lang, use_cuda, audio_data_dict):
    asr = AsrQuartznet(lang=lang, use_cuda=use_cuda)
    audio_data = audio_data_dict[lang]
    text = asr(audio_data)
    assert (type(text) == str)
    assert (len(text) > 0)
    print("Text: {}".format(text))
