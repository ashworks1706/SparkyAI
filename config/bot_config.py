from config.app_config import AppConfig

from utils.common_imports import *


class BotConfig:
    """Configuration for Discord bot"""
    def __init__(self, token: str, app_config: AppConfig):
        self.command_name: str = "ask"
        self.command_description: str = "Ask a question about ASU"
        self.max_question_length: int = 300
        self.max_response_length: int = 2000
        self.chunk_size: int = 1900
        self.token: str = token
        self.thinking_timeout: int = 60
        self.app_config: AppConfig = app_config
