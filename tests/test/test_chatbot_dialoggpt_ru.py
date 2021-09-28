from tgchatbot.text.chatbot_dialoggpt_ru import ChatbotDialoggptRu
import pytest


@pytest.mark.parametrize("use_cuda", [False, True])
def test_chatbot_dialoggpt_ru(use_cuda):
    chat_bot = ChatbotDialoggptRu(use_cuda=use_cuda)
    questions = (
        "Привет! Как дела?",
        "Как тебя зовут?",
        "Какие планы на вечер?",
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
