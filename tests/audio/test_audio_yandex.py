from tgchatbot.audio.audio_yandex import AudioYandex


def test_audio_yandex(pytestconfig):
    oauth_token = pytestconfig.getoption("yandex_oauth_token")
    folder_id = pytestconfig.getoption("yandex_folder_id")
    iam_token = pytestconfig.getoption("yandex_iam_token")

    if oauth_token is not None:
        new_iam_token, expires_at = AudioYandex.create_iam_token(oauth_token)
        assert (new_iam_token is not None)
        assert (type(new_iam_token) == str)
        assert (len(new_iam_token) > 0)
        assert (expires_at is not None)
        assert (type(expires_at) == str)
        assert (len(expires_at) > 0)
        if iam_token is None:
            iam_token = new_iam_token

    if (iam_token is not None) and (folder_id is not None):
        is_valid = AudioYandex.is_iam_token_valid(iam_token=iam_token, folder_id=folder_id)
        assert (type(is_valid) == bool)
