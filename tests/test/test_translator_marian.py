from tgchatbot.text.translator_marian import TranslatorMarian
import pytest


@pytest.mark.parametrize("use_cuda", [False, True])
def test_translator_marian(use_cuda):
    translator = TranslatorMarian(src="ru", dst="fr", use_cuda=use_cuda)
    src_utterances = (
        "Привет! Как дела?",
        "Как тебя зовут?",
        "Какие планы на вечер?",
    )
    for src_utterance in src_utterances:
        dst_utterance = translator(src_utterance)
        assert (type(dst_utterance) == str)
        assert (len(dst_utterance) > 0)
        print("\nQ: {}\nA: {}".format(src_utterance, dst_utterance))
