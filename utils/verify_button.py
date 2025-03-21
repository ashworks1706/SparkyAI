
from utils.common_imports import *
from utils.verification_modal import VerificationModal
class VerifyButton(discord.ui.Button): 
    def __init__(self,config):
        self.config=config
        super().__init__(label="Verify", style=discord.ButtonStyle.primary, custom_id="verify_button")

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.send_modal(VerificationModal(self.config)) 