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
        self.model = EncoderClassifier.from_hparams(source=model_name)

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


if __name__ == "__main__":
    import os.path
    import librosa
    use_cuda = False

    root_dir_path = "../../tgchatbot_data/audio"
    file_paths = {
        "en": os.path.join(root_dir_path, "en/common_voice_en_1.mp3"),
        "fr": os.path.join(root_dir_path, "fr/common_voice_fr_17299384.mp3"),
        "de": os.path.join(root_dir_path, "de/common_voice_de_17298952.mp3"),
        "ru": os.path.join(root_dir_path, "ru/common_voice_ru_18849003.mp3"),
    }

    for lang in file_paths:
        audio_data, sample_rate = librosa.load(path=file_paths[lang], sr=16000, mono=True)
        net = LangidEcapa()
        pred_lang = net(audio_data, sample_rate)
        print("Lang: {}->{}".format(lang, pred_lang))
