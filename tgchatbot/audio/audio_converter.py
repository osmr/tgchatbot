"""
    Simple audio in-memory converter.
"""

__all__ = ['AudioConverter']

from io import BytesIO
import numpy as np
import soundfile as sf
from librosa.core import resample as lr_resample
from pydub import AudioSegment


class AudioConverter(object):
    """
    Simple audio in-memory converter.
    """

    @staticmethod
    def read_from_buffer(audio_buffer: BytesIO,
                         desired_audio_sample_rate: int = 16000) -> np.array:
        """
        Read audio from buffer (in-memory file).

        Parameters:
        ----------
        audio_buffer : io.BytesIO
            Buffer with source audio data.
        desired_audio_sample_rate : int, default 16000
            Desired audio sample rate.

        Returns:
        -------
        np.array
            Audio array.
        """
        audio_segm = AudioSegment.from_file(audio_buffer)
        audio_array = np.array(audio_segm.get_array_of_samples(), dtype=np.float32).reshape(
            (-1, audio_segm.channels)).T / (1 << (8 * audio_segm.sample_width - 1))
        sample_rate = audio_segm.frame_rate
        if audio_array.shape[0] == 1:
            audio_array = audio_array[0]
        if desired_audio_sample_rate != sample_rate:
            audio_array = lr_resample(y=audio_array, orig_sr=sample_rate, target_sr=desired_audio_sample_rate)
        if audio_array.ndim >= 2:
            audio_array = np.mean(audio_array, axis=1)
        return audio_array

    @staticmethod
    def write_to_buffer(audio_array: np.array,
                        audio_sample_rate: int,
                        format: str = "wav") -> BytesIO:
        """
        Write audio to buffer in some format.

        Parameters:
        ----------
        audio : np.array
            Audio data.
        audio_sample_rate : int
            Audio sample rate.
        format : str, default 'wav'
            Encoding format.

        Returns:
        -------
        io.BytesIO
            Output audio buffer.
        """
        audio_buffer = BytesIO()
        sf.write(
            file=audio_buffer,
            data=audio_array,
            samplerate=audio_sample_rate,
            format=format,
            subtype=("PCM_16" if format == "wav" else None))
        audio_buffer.seek(0)
        return audio_buffer
