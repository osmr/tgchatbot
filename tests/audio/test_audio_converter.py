from tgchatbot.audio.audio_converter import AudioConverter
from tgchatbot.audio.asr_quartznet import AsrQuartznet
from io import BytesIO
import numpy as np
import librosa


def test_audio_converter(audio_file_path_dict, use_cuda, tmp_path):
    sample_rate = 16000
    src_audio_file_path = audio_file_path_dict["en"]
    with open(src_audio_file_path, "rb") as f:
        src_buffer = BytesIO(f.read())
    audio_array = AudioConverter.read_from_buffer(
        audio_buffer=src_buffer,
        desired_audio_sample_rate=sample_rate)
    assert (type(audio_array) == np.ndarray)
    assert (len(audio_array) > 0)
    dst_buffer = AudioConverter.write_to_wav_buffer(
        audio_array=audio_array,
        audio_sample_rate=sample_rate)
    assert (isinstance(dst_buffer, BytesIO))
    dst_audio_file_path = tmp_path / "audio_conv_tmp.wav"
    with open(dst_audio_file_path, "wb") as f:
        f.write(dst_buffer.getvalue())

    asr_audio_data = librosa.load(path=dst_audio_file_path, sr=sample_rate, mono=True)[0]
    asr = AsrQuartznet(lang="en", use_cuda=use_cuda)
    src_text = asr(audio_array)
    dst_text = asr(asr_audio_data)
    assert (src_text == dst_text)
    print("Src text: {}".format(src_text))
    print("Dst text: {}".format(dst_text))
