from tgchatbot.text.punct_nemo import PunctNemo
import pytest


@pytest.mark.parametrize("model_type", ["bert", "distilbert"])
@pytest.mark.parametrize("use_cuda", [False, True])
def test_punct_nemo(model_type, use_cuda):
    corrector = PunctNemo(model_type, use_cuda=use_cuda)
    src_utterances = (
        "how are you",
        "great how about you",
    )
    for src_utterance in src_utterances:
        dst_utterance = corrector(src_utterance)
        assert (type(dst_utterance) == str)
        assert (len(dst_utterance) > 0)
        print("\nQ: {}\nA: {}".format(src_utterance, dst_utterance))
