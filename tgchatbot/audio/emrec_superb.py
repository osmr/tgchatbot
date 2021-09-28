"""
    Emotion recognition based on Hubert/Wav2Vec2 for SUPERB Emotion Recognition task.
"""

__all__ = ['EmrecSuperb']


class EmrecSuperb(object):
    """
    Emotion recognition based on Hubert/Wav2Vec2 for SUPERB Emotion Recognition task.

    Parameters:
    ----------
    model_type : str, default 'hubert'
        Model type.
    """
    def __init__(self,
                 model_type="hubert"):
        super(EmrecSuperb, self).__init__()
        assert model_type in ("hubert", "wav2vec2")

        model_name = "superb/{}-base-superb-er".format(model_type)

        from transformers import Wav2Vec2FeatureExtractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        from transformers import HubertForSequenceClassification, Wav2Vec2ForSequenceClassification
        model_class_dict = {"hubert": HubertForSequenceClassification, "wav2vec2": Wav2Vec2ForSequenceClassification}
        self.model = model_class_dict[model_type].from_pretrained(model_name)

    def __call__(self, audio):
        """
        Process an utterance.

        Parameters:
        ----------
        audio : np.array
            Source audio.

        Returns:
        -------
        str
            Emotion label.
        """
        import torch
        inputs = self.feature_extractor(audio, sampling_rate=16000, padding=True, return_tensors="pt")
        logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        labels = [self.model.config.id2label[_id] for _id in predicted_ids.tolist()][0]
        return labels
