from tgchatbot.text.chatbot_dialoggpt_en_multi_wrapper import ChatbotDialoggptEnMultiWrapper
import pytest


@pytest.mark.parametrize("use_cuda", [False, True])
def test_chatbot_dialoggpt_en_multi_wrapper(use_cuda):
    chat_bot = ChatbotDialoggptEnMultiWrapper(lang="de", use_cuda=use_cuda)
    questions = (
        "Hallo! Wie geht es Ihnen?",
        "Wie heißen Sie?",
        "Was sind Ihre Pläne für den Abend?",
    )
    context = [""]
    for question in questions:
        answer = chat_bot(question, context)
        assert (type(answer) == str)
        assert (len(answer) > 0)
        print("\nQ: {}\nA: {}".format(question, answer))
    print("\nContext: {}".format(context))
    assert (type(context) == list)
    assert (len(context) == 1)
    assert (type(context[0]) == str)
    assert (len(context[0]) > 0)
