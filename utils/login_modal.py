# utils/login_modal.py

import discord
from utils.common_imports import *

class LoginModal(discord.ui.Modal):
    """
    A Modal to collect ASUrite credentials; only enforces
    “must be in ACM server” when run in a guild channel.
    """

    def __init__(self, bot):
        super().__init__(title="ASU Login")
        self.bot = bot

        self.asurite_input = discord.ui.TextInput(
            label="ASUrite ID",
            placeholder="Enter your ASUrite ID",
            required=True
        )
        self.add_item(self.asurite_input)

        self.password_input = discord.ui.TextInput(
            label="Password",
            placeholder="Enter your password",
            required=True,
            style=discord.TextStyle.short
        )
        self.add_item(self.password_input)

    async def on_submit(self, interaction: discord.Interaction):
        user_id    = interaction.user.id
        asurite_id = self.asurite_input.value
        password   = self.password_input.value

        # Only enforce server‑membership if this modal was invoked in a guild
        if interaction.guild is not None:
            target_guild = self.bot.client.get_guild(
                int(self.bot.app_config.get_discord_target_guild_id())
            )
            if not target_guild:
                await interaction.response.send_message(
                    "Error: Could not find the ACM Discord server.",
                    ephemeral=True
                )
                return

            try:
                member = await target_guild.fetch_member(user_id)
                if not member:
                    await interaction.response.send_message(
                        "You are not part of the ACM Discord Server. Access is restricted.",
                        ephemeral=True
                    )
                    return
            except discord.NotFound:
                await interaction.response.send_message(
                    "You are not part of the ACM Discord Server. Access is restricted.",
                    ephemeral=True
                )
                return

        # At this point we’re either in DMs or the user passed the guild check
        await interaction.response.defer(thinking=True)
        task_message = await interaction.edit_original_response(
            content="🔐 Starting the login process…"
        )
        await self.bot.utils.start_animation(task_message)

        success = await self.bot.asu_scraper.login_user_credentials(
            user_id, asurite_id, password
        )

        if success:
            await self.bot.utils.stop_animation(
                task_message, "✅ Successfully logged in to MYASU!"
            )
        else:
            await self.bot.utils.stop_animation(
                task_message, "❌ Login failed. Please check your credentials."
            )

    async def on_error(self, interaction: discord.Interaction, error: Exception):
        self.bot.logger.error(f"@discord_bot.py Modal error: {error}", exc_info=True)
        # If something unexpected happened, let the user know
        if not interaction.response.is_done():
            await interaction.response.send_message(
                "An unexpected error occurred. Please try again later.",
                ephemeral=True
            )
