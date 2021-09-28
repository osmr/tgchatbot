"""
    MultiEn/Microsoft's DialogGPT chatbot (Conversational NLU).
"""

__all__ = ['ChatbotDialoggptEnMultiWrapper']

from .chatbot_dialoggpt_en import ChatbotDialoggptEn
from .translator_marian import TranslatorMarian


class ChatbotDialoggptEnMultiWrapper(ChatbotDialoggptEn):
    """
    MultiEn/Microsoft's DialogGPT chatbot (multilingual via English).

    Parameters:
    ----------
    lang : str
        Target language.
    """
    def __init__(self,
                 lang,
                 **kwargs):
        super(ChatbotDialoggptEnMultiWrapper, self).__init__(**kwargs)
        self.lang = lang
        self.target_to_en_translator = TranslatorMarian(src=lang, dst="en", use_cuda=self.use_cuda)
        self.en_to_target_translator = TranslatorMarian(src="en", dst=lang, use_cuda=self.use_cuda)

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
        input_message = self.target_to_en_translator(input_message)
        answer = super(ChatbotDialoggptEnMultiWrapper, self).__call__(input_message, context)
        answer = self.en_to_target_translator(answer)
        return answer
