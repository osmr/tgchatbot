"""
    Multilingual NLU translator (Translation NLU).
"""

__all__ = ['TranslatorMarian']


class TranslatorMarian(object):
    """
    Multilingual NLU translator.

    Parameters:
    ----------
    src : str
        Source language.
    dst : str
        Destination language.
    use_cuda : bool, default False
        Whether to use CUDA.
    """
    def __init__(self,
                 src,
                 dst,
                 use_cuda=False):
        super(TranslatorMarian, self).__init__()
        self.src = src
        self.dst = dst
        self.use_cuda = use_cuda

        model_name = "Helsinki-NLP/opus-mt-{}-{}".format(src, dst)

        from transformers import MarianTokenizer, MarianMTModel
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        assert (not self.model.training)

        if self.use_cuda:
            self.model = self.model.cuda()

    def __call__(self, input_message):
        """
        Process an utterance.

        Parameters:
        ----------
        input_message : str
            Source utterance.

        Returns:
        -------
        str
            Destination utterance.
        """
        input_tokens = self.tokenizer([input_message], return_tensors="pt")
        input_ids = input_tokens["input_ids"]
        attention_mask = input_tokens["attention_mask"]
        if self.use_cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        reply_tokens = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
        answer = self.tokenizer.batch_decode(reply_tokens, skip_special_tokens=True)[0]
        return answer
