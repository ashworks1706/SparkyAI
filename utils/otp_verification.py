class OTPVerificationModal(discord.ui.Modal):
    def __init__(self, correct_otp, email):
        super().__init__(title="Enter OTP")
        self.correct_otp = correct_otp
        self.email = email
        self.otp = discord.ui.TextInput(
            label="ENTER OTP",
            placeholder="Enter the 6-digit OTP sent to your email",
            custom_id="otp_input"
        )
        self.add_item(self.otp)

    async def on_submit(self, interaction: discord.Interaction):
        if self.otp.value == self.correct_otp:
            await self.verify_member(interaction, self.email)
        else:
            await interaction.response.send_message("Incorrect OTP. Please try again.", ephemeral=True)

    async def verify_member(self, interaction: discord.Interaction, email):
        verified_role = discord.utils.get(interaction.guild.roles, name="verified")
        if verified_role:
            await interaction.user.add_roles(verified_role)
            await interaction.response.send_message("You have been verified!", ephemeral=True)
        else:
            await interaction.response.send_message("Verification role not found. Please contact an administrator.", ephemeral=True)

    
