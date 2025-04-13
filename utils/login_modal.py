from utils.verify_button import VerifyButton
from utils.common_imports import *

# Add this class for the login modal
class LoginModal(discord.ui.Modal):
    
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
        user_id = interaction.user.id
        asurite_id = self.asurite_input.value
        password = self.password_input.value
        
        # Check if user is part of the server
        target_guild = self.bot.client.get_guild(self.bot.app_config.get_discord_target_guild_id())
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
                    "You are not part of ACM Discord Server. Access to this command is restricted.",
                    ephemeral=True
                )
                return
                
            # Store the credentials safely in Firestore with user ID
            await self.bot.firestore.login_user_credentials(user_id, asurite_id, password)
            
            await interaction.response.send_message(
                "You have successfully logged in via your ASURite!\nKeep in mind that this is a temporary login and will expire in 6 hours.\nIf you have any issues, please contact the ACM Discord Bot team.",
                ephemeral=True
            )
            
            
            self.bot.logger.info(f"@discord_bot.py User {interaction.user.name} successfully logged in with ASUrite ID")
            
        except discord.NotFound:
            await interaction.response.send_message(
                "You are not part of ACM Discord Server. Access to this command is restricted.",
                ephemeral=True
            )
        except Exception as e:
            self.bot.logger.error(f"@discord_bot.py Error processing login: {str(e)}", exc_info=True)
            await interaction.response.send_message(
                "An error occurred while processing your login. Please try again later.",
                ephemeral=True
            )

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        self.bot.logger.error(f"@discord_bot.py Modal error: {error}", exc_info=True)
        await interaction.response.send_message(
            "An error occurred while processing your login. Please try again later.",
            ephemeral=True
        )
