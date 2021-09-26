"""
    Punctuation and Capitalization NLU based on NeMo.
"""

__all__ = ['PunctNemo']


class PunctNemo(object):
    """
    Punctuation and Capitalization NLU based on NeMo.

    Parameters:
    ----------
    model_type : str, default 'bert'
        Model type.
    use_cuda : bool, default False
        Whether to use CUDA.
    """
    def __init__(self,
                 model_type="bert",
                 use_cuda=False):
        super(PunctNemo, self).__init__()
        assert model_type in ("bert", "distilbert")

        model_name = "punctuation_en_{}".format(model_type)

        from nemo.collections.nlp.models import PunctuationCapitalizationModel
        self.model = PunctuationCapitalizationModel.from_pretrained(model_name)

        if use_cuda:
            self.model = self.model.cuda()

    def __call__(self, text):
        """
        Process an utterance.

        Parameters:
        ----------
        text : str
            Source utterance.

        Returns:
        -------
        str
            Destination utterance.
        """
        corrected_text = self.model.add_punctuation_capitalization([text])
        return corrected_text[0]


if __name__ == "__main__":
    use_cuda = False
    model_types = ("bert", "distilbert")
    for model_type in model_types:
        corrector = PunctNemo(model_type, use_cuda=use_cuda)
        src_utterances = (
            "how are you",
            "great how about you",
        )
        for src_utterance in src_utterances:
            print("\nQ: {}\nA: {}".format(src_utterance, corrector(src_utterance)))
