
@dataclass
class BotConfig:
    """Configuration for Discord bot"""
    command_name: str = "ask"
    command_description: str = "Ask a question about ASU"
    max_question_length: int = 300
    max_response_length: int = 2000
    chunk_size: int = 1900
    token: str = app_config.get_discord_bot_token()  
    thinking_timeout: int = 60

