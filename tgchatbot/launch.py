"""
    Telegram AI chatbot launcher.
"""

__all__ = ['launch_chatbot']

import argparse
import logging
from aiogram import executor
from .telegram_ai_chatbot import TelegramAiChatbot


def launch_chatbot():
    """
    Telegram AI chatbot launch script.
    """
    parser = argparse.ArgumentParser(
        description="Telegram AI chatbot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--token", type=str, required=True, help="Telegram Bot API token")
    parser.add_argument("--use-cuda", action="store_true", help="use CUDA")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    telegram_chat_bot = TelegramAiChatbot(api_token=args.token, use_cuda=args.use_cuda)
    executor.start_polling(telegram_chat_bot.dp, skip_updates=True)


if __name__ == "__main__":
    launch_chatbot()
