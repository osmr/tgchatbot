from tgchatbot.audio.tts_nemo import TtsNemo
from tgchatbot.audio.asr_quartznet import AsrQuartznet
import soundfile as sf
import librosa
import pytest


# @pytest.mark.parametrize("tts_name", ["tacotron2", "glowtts", "fastspeech2", "fastpitch"])
# @pytest.mark.parametrize("vocoder_name", ["waveglow", "squeezewave", "uniglow", "melgan", "hifigan"])
@pytest.mark.parametrize("tts_name", ["tacotron2", "glowtts"])
@pytest.mark.parametrize("vocoder_name", ["waveglow", "squeezewave"])
def test_tts_nemo(tts_name, vocoder_name, use_cuda, tmp_path):
    text = "Recent research at Harvard has shown meditating for as little as 8 weeks, can actually increase the grey " \
           "matter in the parts of the brain responsible for emotional regulation, and learning."
    tts_sample_rate = 22050
    model = TtsNemo(tts_name=tts_name, vocoder_name=vocoder_name, use_cuda=use_cuda)
    audio_data = model(text)
    assert (len(audio_data) > 0)
    audio_file_path = tmp_path / "audio_tts_nemo_{}_{}.wav".format(tts_name, vocoder_name)
    sf.write(
        file=audio_file_path,
        data=audio_data,
        samplerate=tts_sample_rate,
        subtype="PCM_16")

    asr_sample_rate = 16000
    asr_audio_data = librosa.load(path=audio_file_path, sr=asr_sample_rate, mono=True)[0]
    asr = AsrQuartznet(lang="en", use_cuda=use_cuda)
    asr_text = asr(asr_audio_data)
    assert (type(asr_text) == str)
    assert (len(asr_text) > 0)
    assert (0.5 < float(len(asr_text)) / len(text) < 2.0)
    print("Src text: {}".format(text))
    print("Dst text: {}".format(asr_text))
