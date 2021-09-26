"""
    Abstract chatbot class description (Conversational NLU).
"""

__all__ = ['Chatbot']


class Chatbot(object):
    """
    Abstract chatbot class.

    Parameters:
    ----------
    use_cuda : bool, default False
        Whether to use CUDA.
    """
    def __init__(self,
                 use_cuda=False):
        super(Chatbot, self).__init__()
        self.use_cuda = use_cuda

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
        """
        raise NotImplementedError
