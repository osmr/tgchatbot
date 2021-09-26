"""
    FR/Microsoft's DialogGPT chatbot (Conversational NLU).
"""

__all__ = ['ChatbotDialoggptFr']

from .chatbot import Chatbot


class ChatbotDialoggptFr(Chatbot):
    """
    FR/Microsoft's DialogGPT chatbot.
    """
    def __init__(self,
                 **kwargs):
        super(ChatbotDialoggptFr, self).__init__(**kwargs)
        model_name = "cedpsam/chatbot_fr"

        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        assert (not self.model.training)

        if self.use_cuda:
            self.model = self.model.cuda()

    def _decorate_utterance(self, input_message):
        return "{}{}".format(input_message, self.tokenizer.eos_token)

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

        input_tokens = self.tokenizer.encode(context[0], return_tensors="pt")
        if self.use_cuda:
            input_tokens = input_tokens.cuda()

        reply_tokens = self.model.generate(
            input_ids=input_tokens,
            num_return_sequences=1,
            max_length=256,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.6,
            mask_token_id=self.tokenizer.mask_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            unk_token_id=self.tokenizer.unk_token_id,
            pad_token_id=self.tokenizer.eos_token_id)
        answer = self.tokenizer.batch_decode(reply_tokens[:, input_tokens.shape[-1]:], skip_special_tokens=True)[0]

        context[0] += self._decorate_utterance(answer)

        return answer


if __name__ == "__main__":
    use_cuda = False
    chat_bot = ChatbotDialoggptFr(use_cuda=use_cuda)
    questions = (
        "Salut! Comment ca va?",
        "Quel est ton nom?",
        "Que comptes-tu faire ce soir?",
    )
    context = [""]
    for question in questions:
        print("\nQ: {}\nA: {}".format(question, chat_bot(question, context)))
    print("\nContext: {}".format(context))
