"""
    EN/Facebook's Blenderbot chatbot (Conversational NLU).
"""

__all__ = ['ChatbotBlenderbotEn']

from .chatbot import Chatbot


class ChatbotBlenderbotEn(Chatbot):
    """
    EN/Facebook's Blenderbot chatbot.
    """
    def __init__(self,
                 **kwargs):
        super(ChatbotBlenderbotEn, self).__init__(**kwargs)
        model_name = "facebook/blenderbot-400M-distill"

        from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
        assert (not self.model.training)

        if self.use_cuda:
            self.model = self.model.cuda()

    @staticmethod
    def _decorate_utterance(input_message):
        return "<s>{}</s>".format(input_message)

    def __call__(self,
                 input_message,
                 context=None):
        """
        Process a question.

        Parameters:
        ----------
        input_message : str
            Question.
        context : list of str, default None
            History of conversation.

        Returns:
        -------
        str
            Answer.
        """
        assert (context is None) or ((type(context) is list) and (len(context) == 1) and (type(context[0]) is str))

        if context is None:
            context = [""]

        input_message = self._decorate_utterance(input_message)

        context[0] += input_message

        input_tokens = self.tokenizer([context[0]], return_tensors="pt")
        input_ids = input_tokens["input_ids"]
        attention_mask = input_tokens["attention_mask"]
        if self.use_cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        reply_tokens = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
        answer = self.tokenizer.batch_decode(reply_tokens, skip_special_tokens=True)[0]

        context[0] += self._decorate_utterance(answer)

        return answer
