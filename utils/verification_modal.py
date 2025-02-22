from utils.common_imports import *
class VerificationModal(discord.ui.Modal):
    def __init__(self):
        super().__init__(title="ASU Email Verification")
        self.email = discord.ui.TextInput(
            label="ENTER YOUR ASU EMAIL",
            placeholder="ASURITE@asu.edu",
            custom_id="email_input"
        )
        self.stored_otp = None
        self.add_item(self.email)

    async def on_submit(self, interaction: discord.Interaction):
        if  not self.stored_otp:  # First submission - email only
            if self.validate_asu_email(self.email.value):
                self.stored_otp = self.generate_otp()
                self.send_otp_email(self.email.value, self.stored_otp)
                view = discord.ui.View()
                button = discord.ui.Button(label="Enter OTP", style=discord.ButtonStyle.primary)
                async def button_callback(button_interaction):
                    await button_interaction.response.send_modal(OTPVerificationModal(self.stored_otp, self.email.value))
                button.callback = button_callback
                view.add_item(button)
                await interaction.response.send_message("OTP has been sent to your email. Click the button below to enter it.", view=view, ephemeral=True)
            else:
                await interaction.response.send_message("Invalid ASU email. Please try again.", ephemeral=True)

    def validate_asu_email(self, email):
        return re.match(r'^[a-zA-Z0-9._%+-]+@asu\.edu$', email) is not None

    def generate_otp(self):
        return ''.join(str(random.randint(0, 9)) for _ in range(6))

    def send_otp_email(self, email, otp):
        sender_email = app_config.get_gmail()
        sender_password = app_config.get_gmail_pass()
        message = MIMEText(f"Your OTP for ASU Discord verification is {otp}")
        message['Subject'] = "ASU Discord Verification OTP"
        message['From'] = sender_email
        message['To'] = email
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message.as_string())
          