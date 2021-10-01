from pathlib import Path
import librosa
import pytest


@pytest.fixture(scope="module")
def audio_file_path_dict():
    data_dir_path = Path(__file__).parents[2].absolute() / "data"
    audio_file_path_dict = {
        "en": "common_voice_en_1.mp3",
        "fr": "common_voice_fr_17299384.mp3",
        "de": "common_voice_de_17298952.mp3",
        "ru": "common_voice_ru_18849003.mp3",
    }
    audio_file_path_dict = {lang: str(data_dir_path / audio_file_path_dict[lang])
                            for lang in audio_file_path_dict}
    return audio_file_path_dict


@pytest.fixture(scope="module")
def audio_data_dict(audio_file_path_dict):
    sample_rate = 16000
    audio_data_dict = {lang: librosa.load(path=audio_file_path_dict[lang], sr=sample_rate, mono=True)[0]
                       for lang in audio_file_path_dict}
    return audio_data_dict


def pytest_addoption(parser):
    parser.addoption("--yandex-oauth-token", action="store", default=None, help="Yandex OAuth token")
    parser.addoption("--yandex-folder-id", action="store", default=None, help="Yandex Cloud folder id")
    parser.addoption("--yandex-iam-token", action="store", default=None, help="Yandex IAM token")
    parser.addoption("--google-credentials", action="store", default=None, help="Google app credentials")
