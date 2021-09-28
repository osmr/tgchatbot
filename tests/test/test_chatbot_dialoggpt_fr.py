from tgchatbot.text.chatbot_dialoggpt_fr import ChatbotDialoggptFr
import pytest


@pytest.mark.parametrize("use_cuda", [False, True])
def test_chatbot_dialoggpt_fr(use_cuda):
    chat_bot = ChatbotDialoggptFr(use_cuda=use_cuda)
    questions = (
        "Salut! Comment ca va?",
        "Quel est ton nom?",
        "Que comptes-tu faire ce soir?",
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
