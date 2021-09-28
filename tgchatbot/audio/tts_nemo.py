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
