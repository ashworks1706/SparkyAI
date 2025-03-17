import logging
from utils.common_imports import *
from utils.otp_verification import OTPVerificationModal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VerificationModal(discord.ui.Modal):
    def __init__(self, config):
        super().__init__(title="ASU Email Verification")
        self.email = discord.ui.TextInput(
            label="ENTER YOUR ASU EMAIL",
            placeholder="ASURITE@asu.edu",
            custom_id="email_input"
        )
        self.app_config = config
        self.stored_otp = None
        self.add_item(self.email)
        logger.info("VerificationModal initialized with config: %s", config)

    async def on_submit(self, interaction: discord.Interaction):
        logger.info("Form submitted with email: %s", self.email.value)
        if not self.stored_otp:  # First submission - email only
            if self.validate_asu_email(self.email.value):
                self.stored_otp = self.generate_otp()
                logger.info("Generated OTP: %s", self.stored_otp)
                self.send_otp_email(self.email.value, self.stored_otp)
                view = discord.ui.View()
                button = discord.ui.Button(label="Enter OTP", style=discord.ButtonStyle.primary)
                
                async def button_callback(button_interaction):
                    logger.info("OTP button clicked for email: %s", self.email.value)
                    await button_interaction.response.send_modal(OTPVerificationModal(self.stored_otp, self.email.value))
                
                button.callback = button_callback
                view.add_item(button)
                await interaction.response.send_message("OTP has been sent to your email. Click the button below to enter it.", view=view, ephemeral=True)
                logger.info("OTP sent to email: %s", self.email.value)
            else:
                await interaction.response.send_message("Invalid ASU email. Please try again.", ephemeral=True)
                logger.warning("Invalid ASU email entered: %s", self.email.value)

    def validate_asu_email(self, email):
        is_valid = re.match(r'^[a-zA-Z0-9._%+-]+@asu\.edu$', email) is not None
        logger.info("Email validation for %s: %s", email, is_valid)
        return is_valid

    def generate_otp(self):
        otp = ''.join(str(random.randint(0, 9)) for _ in range(6))
        logger.info("Generated OTP: %s", otp)
        return otp

    def send_otp_email(self, email, otp):
        sender_email = self.app_config.get_gmail()
        sender_password = self.app_config.get_gmail_pass()
        message = MIMEText(f"Your OTP for ASU Discord verification is {otp}")
        message['Subject'] = "ASU Discord Verification OTP"
        message['From'] = sender_email
        message['To'] = email
        
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, email, message.as_string())
            logger.info("OTP email sent to %s", email)
        except Exception as e:
            logger.error("Failed to send OTP email to %s: %s", email, str(e))