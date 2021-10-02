"""
    ASR from Amazon Cloud (Transcribe).
"""

__all__ = ['AsrAmazon']

from io import BytesIO
import asyncio


class AsrAmazon(object):
    """
    ASR from Amazon Cloud (Transcribe).
    See https://docs.aws.amazon.com/transcribe/latest/dg/transcribe-whatis.html and
    https://github.com/awslabs/amazon-transcribe-streaming-sdk.

    Parameters:
    ----------
    lang : str
        Language.
    aws_access_key_id : str
        AWS access key ID.
    aws_secret_access_key : str
        AWS secret access key.
    aws_region_name : str
        AWS region when creating new connections.
    use_cuda : bool, default False
        Whether to use CUDA (fake argument for a Cloud Service).
    """
    def __init__(self,
                 lang,
                 aws_access_key_id,
                 aws_secret_access_key,
                 aws_region_name,
                 use_cuda=False):
        super(AsrAmazon, self).__init__()
        assert (lang in ("en", "fr", "de"))
        self.lang = lang
        self.use_cuda = use_cuda
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key

        lang_code_dict = {"en": "en-US", "fr": "fr-FR", "de": "de-DE", "ru": "ru-RU"}
        self.lang_code = lang_code_dict[lang]

        from amazon_transcribe.client import TranscribeStreamingClient
        self.client = TranscribeStreamingClient(region=aws_region_name)

    def __call__(self, input_audio):
        """
        Process an utterance.

        Parameters:
        ----------
        input_audio : np.array
            Source audio.

        Returns:
        -------
        str
            Destination text.
        """
        from amazon_transcribe.handlers import TranscriptResultStreamHandler
        from amazon_transcribe.model import TranscriptEvent

        text_list = []

        class MyEventHandler(TranscriptResultStreamHandler):
            async def handle_transcript_event(self, transcript_event: TranscriptEvent):
                results = transcript_event.transcript.results
                for result in results:
                    for alt in result.alternatives:
                        text_list.append(alt.transcript)

        self.audio_buffer = BytesIO(input_audio.tobytes())
        self.audio_buffer.seek(0)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.transcribe(MyEventHandler))
        loop.close()

        text = "".join(text_list)
        return text

    async def transcribe(self, hendler_class):
        """
        Internal rountine for transcribing.
        """
        import aiofile

        stream = await self.client.start_stream_transcription(
            language_code=self.lang_code,
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )

        async def write_chunks():
            async with aiofile.AIOFile(self.audio_buffer, "rb") as afp:
                reader = aiofile.Reader(afp, chunk_size=1024 * 16)
                async for chunk in reader:
                    await stream.input_stream.send_audio_event(audio_chunk=chunk)
            await stream.input_stream.end_stream()

        # Instantiate our handler and start processing events
        handler = hendler_class(stream.output_stream)
        await asyncio.gather(write_chunks(), handler.handle_events())
