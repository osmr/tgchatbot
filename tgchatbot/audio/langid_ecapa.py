"""
    Language identification based on ECAPA-TDNN.
"""

__all__ = ['LangidEcapa']


class LangidEcapa(object):
    """
    Language identification based on ECAPA-TDNN.
    """
    def __init__(self):
        super(LangidEcapa, self).__init__()

        from speechbrain.pretrained import EncoderClassifier
        # model_name = "TalTechNLP/voxlingua107-epaca-tdnn"
        model_name = "TalTechNLP/voxlingua107-epaca-tdnn-ce"
        self.model = EncoderClassifier.from_hparams(source=model_name, savedir="/tmp")

    def __call__(self, audio, sample_rate):
        """
        Process an utterance.

        Parameters:
        ----------
        audio : np.array
            Source audio.
        sample_rate : int
            Audio sample rate.

        Returns:
        -------
        str
            Language label.
        """
        import torch
        audio = torch.from_numpy(audio)
        audio = self.model.audio_normalizer(audio, sample_rate)
        prediction = self.model.classify_batch(audio)
        return prediction[3][0]
