from tgchatbot.text.chatbot_blenderbot_en import ChatbotBlenderbotEn
import pytest


@pytest.mark.parametrize("use_cuda", [False, True])
def test_chatbot_blenderbot_en(use_cuda):
    chat_bot = ChatbotBlenderbotEn(use_cuda=use_cuda)
    questions = (
        "Hello! How are you?",
        "What's your name?",
        "What are your plans for the evening?",
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
