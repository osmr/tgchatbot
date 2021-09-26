"""
    TTS from NeMo.
"""

__all__ = ['TtsNemo']


class TtsNemo(object):
    """
    TTS from NeMo for English.

    Parameters:
    ----------
    tts_name : str, default 'tacotron2'
        TTS model name.
    vocoder_name : str, default 'waveglow'
        TTS model name.
    use_cuda : bool, default False
        Whether to use CUDA.
    """
    def __init__(self,
                 tts_name="tacotron2",
                 vocoder_name="waveglow",
                 use_cuda=False):
        super(TtsNemo, self).__init__()
        assert (tts_name in ("tacotron2", "glowtts", "fastspeech2", "fastpitch"))
        assert (vocoder_name in ("waveglow", "squeezewave", "uniglow", "melgan", "hifigan"))

        if tts_name == "tacotron2":
            from nemo.collections.tts.models import Tacotron2Model
            self.tts = Tacotron2Model.from_pretrained(model_name="tts_en_tacotron2")
        elif tts_name == "glowtts":
            from nemo.collections.tts.models import GlowTTSModel
            self.tts = GlowTTSModel.from_pretrained(model_name="tts_en_glowtts")
        elif tts_name == "fastspeech2":
            from nemo.collections.tts.models import FastSpeech2Model
            self.tts = FastSpeech2Model.from_pretrained(model_name="tts_en_fastspeech2")
        elif tts_name == "fastpitch":
            from nemo.collections.tts.models import FastPitchModel
            self.tts = FastPitchModel.from_pretrained(model_name="tts_en_fastpitch")
        else:
            raise ValueError("Unsupported TTS model: {}".format(tts_name))

        if vocoder_name == "waveglow":
            from nemo.collections.tts.models import WaveGlowModel
            self.vocoder = WaveGlowModel.from_pretrained(model_name="tts_waveglow")
        elif vocoder_name == "squeezewave":
            from nemo.collections.tts.models import SqueezeWaveModel
            self.vocoder = SqueezeWaveModel.from_pretrained(model_name="tts_squeezewave")
        elif vocoder_name == "uniglow":
            from nemo.collections.tts.models import UniGlowModel
            self.vocoder = UniGlowModel.from_pretrained(model_name="tts_uniglow")
        elif vocoder_name == "melgan":
            from nemo.collections.tts.models import MelGanModel
            self.vocoder = MelGanModel.from_pretrained(model_name="tts_melgan")
        elif vocoder_name == "hifigan":
            from nemo.collections.tts.models import HifiGanModel
            self.vocoder = HifiGanModel.from_pretrained(model_name="tts_hifigan")
        else:
            raise ValueError("Unsupported vocoder model: {}".format(vocoder_name))

        if use_cuda:
            self.tts = self.tts.cuda()
            self.vocoder = self.vocoder.cuda()

    def __call__(self, text):
        """
        Process an utterance.

        Parameters:
        ----------
        text : str
            Source utterance.

        Returns:
        -------
        np.array
            Destination audio.
        """
        import torch
        with torch.no_grad():
            input_tokens = self.tts.parse(text)
            spectrogram = self.tts.generate_spectrogram(tokens=input_tokens)
            audio = self.vocoder.convert_spectrogram_to_audio(spec=spectrogram)
            audio = audio[0].cpu().detach().numpy()
        return audio


if __name__ == "__main__":
    import os.path
    import soundfile as sf
    use_cuda = False

    root_dir_path = "../../tgchatbot_data/audio"
    text = "Recent research at Harvard has shown meditating for as little as 8 weeks, can actually increase the grey " \
           "matter in the parts of the brain responsible for emotional regulation, and learning."
    model_params = (
        {"tts_name": "tacotron2", "vocoder_name": "waveglow"},
        {"tts_name": "glowtts", "vocoder_name": "waveglow"},
        {"tts_name": "tacotron2", "vocoder_name": "squeezewave"},
        {"tts_name": "glowtts", "vocoder_name": "squeezewave"},
    )

    for model_param in model_params:
        tts_name = model_param["tts_name"]
        vocoder_name = model_param["vocoder_name"]
        model1 = TtsNemo(tts_name=tts_name, vocoder_name=vocoder_name, use_cuda=use_cuda)
        audio = model1(text)
        sf.write(
            file=os.path.join(root_dir_path, "audio_en_nemo_{}_{}.wav".format(tts_name, vocoder_name)),
            data=audio,
            samplerate=22050,
            subtype="PCM_16")
