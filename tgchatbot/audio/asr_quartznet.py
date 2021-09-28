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
