from utils.common_imports import *
class DiscordState:
    def __init__(self):
        nest_asyncio.apply()
        self.intents = discord.Intents.default()
        self.intents.message_content = True
        self.intents.members = True
        self.user = False
        self.target_guild = None
        self.user_id = None
        self.user_has_mod_role = False
        self.user_in_voice_channel = False
        self.request_in_dm = False
        self.guild_user= None
        self.user_voice_channel_id = None
        self.discord_client = discord.Client(intents=self.intents)
        self.task_message = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"DiscordState has no attribute '{key}'")

    def get(self, attr):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            raise AttributeError(f"DiscordState has no attribute '{attr}'")

    def __str__(self):
        return "\n".join([f"{attr}: {getattr(self, attr)}" for attr in vars(self) if not attr.startswith('__')])
