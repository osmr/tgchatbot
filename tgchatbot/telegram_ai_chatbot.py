"""
    Telegram AI chatbot.
"""

__all__ = ['TelegramAiChatbot']

from functools import partial
from io import BytesIO
from aiogram import Bot, Dispatcher, types
from .text.chatbot_blenderbot_en import ChatbotBlenderbotEn
from .text.chatbot_dialoggpt_en_multi_wrapper import ChatbotDialoggptEnMultiWrapper
from .text.chatbot_dialoggpt_fr import ChatbotDialoggptFr
from .text.chatbot_dialoggpt_ru import ChatbotDialoggptRu
from .text.translator_marian import TranslatorMarian
from .audio.audio_converter import AudioConverter


class TelegramAiChatbot(object):
    """
    Telegram AI chatbot.

    Parameters:
    ----------
    api_token : str
        Telegram Bot API token.
    use_cuda : bool, default False
        Whether to use CUDA.
    """
    def __init__(self,
                 api_token,
                 use_cuda=False):
        super(TelegramAiChatbot, self).__init__()
        self.use_cuda = use_cuda
        self.HELP_MESSAGE = "This is a chatbot. Talk to me."

        self.bot = Bot(token=api_token)
        self.dp = Dispatcher(self.bot)

        self.langs = ("en", "de", "fr", "ru")

        self.asr = {lang: None for lang in self.langs}

        self.use_tts = True
        self.tts = {
            "en": None,
            "fr": None,
        }

        self.text_chatbot_classes = {
            "en": ChatbotBlenderbotEn,
            "de": partial(ChatbotDialoggptEnMultiWrapper, lang="de"),
            "fr": ChatbotDialoggptFr,
            "ru": ChatbotDialoggptRu,
        }
        self.text_chatbots = {lang: None for lang in self.langs}
        self.text_chatbot_contexts = {}

        self.lang_states = {}
        self.text_translators = {}

        @self.dp.message_handler(commands=["start"])
        async def process_start_command(message: types.Message):
            """
            Show welcome message. Here you can change both languages by sending the appropriate string with the
            language code.
            """
            user_id = message.from_user.id
            lang_state = self.get_lang_state(user_id)

            command_parts = message.text.split()
            if (len(command_parts) == 2) and ((command_parts[1] in self.langs) and
                                              ((command_parts[1] != lang_state["src"]) or
                                               (command_parts[1] != lang_state["dst"]))):
                lang_state["src"] = command_parts[1]
                lang_state["dst"] = command_parts[1]
            await message.answer(self.HELP_MESSAGE)

        @self.dp.message_handler(commands=["help"])
        async def process_help_command(message: types.Message):
            """
            Show welcome message.
            """
            await message.answer(self.HELP_MESSAGE)

        @self.dp.message_handler(commands=["lang", "lang_src", "lang_dst"])
        async def process_lang_command(message: types.Message):
            """
            Show current languages status. You can change languages by sending the appropriate string with the
            language code.
            """
            user_id = message.from_user.id
            lang_state = self.get_lang_state(user_id)

            command_parts = message.text.split()
            if len(command_parts) == 1:
                if command_parts[0] == "/lang":
                    key = self.get_lang_key(lang_state)
                    await message.answer(text="`Language state:` {}".format(key), parse_mode="Markdown")
                elif command_parts[0] == "/lang_src":
                    await message.answer(
                        text="`Source language state:` {}".format(lang_state["src"]),
                        parse_mode="Markdown")
                else:
                    assert (command_parts[0] == "/lang_dst")
                    await message.answer(
                        text="`Destination language state:` {}".format(lang_state["dst"]),
                        parse_mode="Markdown")
            else:
                if (len(command_parts) != 2) or command_parts[1] not in self.langs:
                    await message.answer(
                        text="Wrong request! Supported languages are `{}`.".format("`, `".join(self.langs)),
                        parse_mode="Markdown")
                else:
                    if command_parts[0] == "/lang":
                        lang_state["src"] = command_parts[1]
                        lang_state["dst"] = command_parts[1]
                    elif command_parts[0] == "/lang_src":
                        lang_state["src"] = command_parts[1]
                    else:
                        assert (command_parts[0] == "/lang_dst")
                        lang_state["dst"] = command_parts[1]
                    key = self.get_lang_key(lang_state)
                    await message.answer(text="`New language state:` {}".format(key), parse_mode="Markdown")

        @self.dp.message_handler(commands=["tts"])
        async def process_tts_command(message: types.Message):
            """
            Show current TTS activity status. You can change this status by sending `yes` or `no` string.
            """
            user_id = message.from_user.id
            lang_state = self.get_lang_state(user_id)

            command_parts = message.text.split()
            if len(command_parts) == 1:
                await message.answer(
                    text="`Enable TTS:` {}".format("yes" if self.use_tts else "no"),
                    parse_mode="Markdown")
            else:
                if (len(command_parts) != 2) or command_parts[1] not in ("yes", "no"):
                    await message.answer(
                        text="Wrong request! Supported states for TTS are `yes` and `no`.",
                        parse_mode="Markdown")
                else:
                    self.use_tts = (command_parts[1] == "yes")
                    if self.use_tts and (lang_state["dst"] not in ("en", "fr")):
                        await message.answer("Actually only English and French are supported for TTS")
                    else:
                        await message.answer(
                            text="`Enable TTS:` {}".format("yes" if self.use_tts else "no"),
                            parse_mode="Markdown")

        @self.dp.message_handler(commands=["context"])
        async def process_context_command(message: types.Message):
            """
            Show contest for text (NLU) chatbot context corresponding the current user and output language. You can
            erase the context by sending empty string.
            """
            user_id = message.from_user.id
            lang_state = self.get_lang_state(user_id)
            nlu_chat_bot_context = self.get_text_chatbot_context(user_id, lang_state["dst"])

            command_parts = message.text.split()
            if len(command_parts) == 1:
                await message.answer(nlu_chat_bot_context[0] if len(nlu_chat_bot_context[0]) > 0 else "...")
            else:
                nlu_chat_bot_context[0] = ""

        @self.dp.message_handler(content_types=types.message.ContentType.TEXT)
        async def process_text_user_message(message: types.Message):
            """
            Response to a text message.
            """
            user_id = message.from_user.id
            text_answer = self.answer_text(text=message.text, user_id=user_id)

            tts = self.get_tts(user_id)
            if tts is not None:
                audio_answer = tts(text_answer)
                audio_sample_rate = 22050
                await message.answer_voice(
                    voice=AudioConverter.write_to_wav_buffer(
                        audio_array=audio_answer,
                        audio_sample_rate=audio_sample_rate),
                    caption=text_answer,
                    duration=(len(audio_answer) // audio_sample_rate)
                )
            else:
                await message.answer(text_answer)

        @self.dp.message_handler(content_types=types.message.ContentType.AUDIO)
        async def process_audio_user_message(message: types.Message):
            """
            Response to a sound file as an audio speech.
            """
            try:
                audio = message.audio
                audio_buffer = BytesIO()
                await self.bot.download_file_by_id(file_id=audio.file_id, destination=audio_buffer)
                audio_data = AudioConverter.read_from_buffer(
                    audio_buffer=audio_buffer,
                    desired_audio_sample_rate=16000)
            except RuntimeError:
                await message.answer("Error reading file")

            user_id = message.from_user.id
            asr = self.get_asr(user_id)
            text_question = asr(audio_data)

            text_answer = self.answer_text(text=text_question, user_id=user_id)

            tts = self.get_tts(user_id)
            if tts is not None:
                audio_answer = tts(text_answer)
                audio_sample_rate = 22050
                await message.answer_voice(
                    voice=AudioConverter.write_to_wav_buffer(
                        audio_array=audio_answer,
                        audio_sample_rate=audio_sample_rate),
                    caption="Q: {}\nA: {}".format(text_question, text_answer),
                    duration=(len(audio_answer) // audio_sample_rate)
                )
            else:
                await message.answer(text_answer)

    def get_lang_state(self, user_id):
        """
        Get language state (pair of languages: input and output) for the session.

        Parameters:
        ----------
        user_id : str
            User ID as session.

        Returns:
        -------
        dict
            Language state.
        """
        if user_id not in self.lang_states:
            self.lang_states[user_id] = {"src": self.langs[0], "dst": self.langs[0]}
        return self.lang_states[user_id]

    def get_asr(self, user_id):
        """
        Get an Automatic Speech Recognition processor.

        Parameters:
        ----------
        user_id : str
            User ID as session.

        Returns:
        -------
        object
            ASR processor.
        """
        lang_state = self.get_lang_state(user_id)
        if self.asr[lang_state["src"]] is None:
            from .audio.asr_quartznet import AsrQuartznet
            self.asr[lang_state["src"]] = AsrQuartznet(lang=lang_state["src"], use_cuda=self.use_cuda)
        return self.asr[lang_state["src"]]

    def get_tts(self, user_id):
        """
        Get a Text-To-Speech processor.

        Parameters:
        ----------
        user_id : str
            User ID as session.

        Returns:
        -------
        object
            TTS processor.
        """
        lang_state = self.get_lang_state(user_id)
        if self.use_tts and (lang_state["dst"] in ("en", "fr")):
            if self.tts[lang_state["dst"]] is None:
                from .audio.tts_tensorspeech import TtsTensorspeech
                self.tts[lang_state["dst"]] = TtsTensorspeech(lang=lang_state["dst"], use_cuda=self.use_cuda)
            return self.tts[lang_state["dst"]]
        else:
            return None

    def get_text_translator(self, lang_state):
        """
        Get a text (NLU) translator corresponding to the pair of languages for the session.

        Parameters:
        ----------
        lang_state : dict
            Language state (pair of languages: input and output).

        Returns:
        -------
        object
            Text (NLU) translator.
        """
        key = self.get_lang_key(lang_state)
        if key not in self.text_translators:
            self.text_translators[key] = TranslatorMarian(
                src=lang_state["src"],
                dst=lang_state["dst"],
                use_cuda=self.use_cuda)
        return self.text_translators[key]

    def get_text_chatbot(self, lang):
        """
        Get a text (NLU) chatbot corresponding to the destination language for the session.

        Parameters:
        ----------
        lang : str
            Language.

        Returns:
        -------
        object
            Text (NLU) chatbot.
        """
        if self.text_chatbots[lang] is None:
            self.text_chatbots[lang] = self.text_chatbot_classes[lang](use_cuda=self.use_cuda)
        return self.text_chatbots[lang]

    def get_text_chatbot_context(self, user_id, lang):
        """
        Get contest for text (NLU) chatbot for the session.

        Parameters:
        ----------
        user_id : str
            User ID as session.
        lang : str
            Language.

        Returns:
        -------
        str
            Context string.
        """
        if user_id not in self.text_chatbot_contexts:
            self.text_chatbot_contexts[user_id] = {lang: [""] for lang in self.langs}
        return self.text_chatbot_contexts[user_id][lang]

    def answer_text(self, text, user_id):
        """
        Text-to-text reply.

        Parameters:
        ----------
        text : str
            Input utterance (or question).
        user_id : str
            User ID as session.

        Returns:
        -------
        str
            Answer or (output utterance).
        """
        lang_state = self.get_lang_state(user_id)

        if lang_state["dst"] != lang_state["src"]:
            nlu_translator = self.get_text_translator(lang_state)
            text = nlu_translator(text)

        answer = self.get_text_chatbot(lang_state["dst"])(
            input_message=text,
            context=self.get_text_chatbot_context(user_id, lang_state["dst"]))

        return answer

    @staticmethod
    def get_lang_key(lang_state):
        """
        Get language key string.

        Parameters:
        ----------
        lang_state : dict
            Language state (pair of languages: input and output).

        Returns:
        -------
        str
            Target key string.
        """
        return "{}-{}".format(lang_state["src"], lang_state["dst"])
