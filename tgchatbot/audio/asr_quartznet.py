"""
    ASR based on QuartzNet.
"""

__all__ = ['AsrQuartznet']


class AsrQuartznet(object):
    """
    ASR based on QuartzNet.

    Parameters:
    ----------
    lang : str
        Language.
    use_cuda : bool, default False
        Whether to use CUDA.
    """
    def __init__(self,
                 lang,
                 use_cuda=False):
        super(AsrQuartznet, self).__init__()
        assert (lang in ("en", "fr", "de", "ru"))
        self.lang = lang
        self.use_cuda = use_cuda

        from pytorchcv.model_provider import get_model as ptcv_get_model

        if lang != "ru":
            asr_model_name = "quartznet15x5_{}".format(lang)
        else:
            asr_model_name = "quartznet15x5_ru34"

        self.net = ptcv_get_model(asr_model_name, return_text=True, pretrained=True)

        if use_cuda:
            self.net = self.net.cuda()

        self.net.eval()

    def __call__(self, input_audio):
        """
        Process an utterance.

        Parameters:
        ----------
        input_audio : np.array
            Source audio.

        Returns:
        -------
        str
            Destination text.
        """
        import torch
        x = torch.from_numpy(input_audio).unsqueeze(0)
        x_len = torch.tensor([input_audio.shape[0]], dtype=torch.long)

        if self.use_cuda:
            x = x.cuda()
            x_len = x_len.cuda()

        text = self.net(x, x_len)[0]
        return text


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
        asr = AsrQuartznet(lang=lang, use_cuda=use_cuda)
        text = asr(audio_data)
        print("Text: {}".format(text))
