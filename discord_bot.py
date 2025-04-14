from utils.verify_button import VerifyButton
from utils.login_modal import LoginModal
from utils.common_imports import * 

class ASUDiscordBot:
    
    """Discord bot for handling ASU-related questions"""

    # Your existing __init__ method, but add the login command
    def __init__(self, config, app_config, agents, firestore, discord_state, utils,vector_store,logger,asu_scraper ):
        """
        Initialize the Discord bot.
        
        Args:
            agents: RAG pipeline instance
            config: Optional bot configuration
        """
        self.logger= logger
        self.logger.info(f"@discord_bot.py  Initializing ASUDiscordBot")
        self.config = config or BotConfig(app_config)
        self.app_config= app_config
        self.agents = agents
        self.firestore = firestore
        self.utils = utils
        self.asu_scraper = asu_scraper
        self.vector_store = vector_store
        
        # Initialize Discord client
        self.discord_state = discord_state
        self.client = discord_state.get('discord_client')
        self.tree = app_commands.CommandTree(self.client)
        
        @self.client.event
        async def on_ready():
            await self._handle_ready()
            
        @self.tree.command(
            name=self.config.command_name,
            description=self.config.command_description
        )
        async def ask(interaction: discord.Interaction, question: str):
            await self._handle_ask_command(interaction, question)
        
        @self.tree.command(
            name="login",
            description="Login with your ASUrite credentials"
        )
        async def login(interaction: discord.Interaction):
            await self._handle_login_command(interaction)
        @self.tree.command(
            name="logout",
            description="Login with your ASUrite credentials"
        )
        async def logout(interaction: discord.Interaction):
            await self._handle_logout_command(interaction)
    
    async def _handle_logout_command(self, interaction: discord.Interaction) -> None:
        """
        Handle the logout command.
        
        Args:
            interaction: Discord interaction
        """
        try:
            
            if self.discord_state.get('user_session_id') == None:
                await interaction.response.send_message(
                    "You are already logged out.",
                    ephemeral=True
                )
                return
            await interaction.response.defer(thinking=True)
            global task_message
            task_message = await interaction.edit_original_response(content="Starting the logout process...")
            
            
            await self.asu_scraper.logout_user_credentials()
            # Reset user session ID and other states
            self.discord_state.update(user_session_id=None)
            
            await self.utils.start_animation(task_message)
            
            self.utils.stop_animation(task_message, "Successfully logged out of MYASU!")
            
            self.logger.info(f"@discord_bot.py User {interaction.user.name} successfully logged out")
        
        except Exception as e:
            error_msg = f"Error processing logout command: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            await self._send_error_response(interaction)
    
    async def _handle_login_command(self, interaction: discord.Interaction) -> None:
        """
        Handle the login command.
        
        Args:
            interaction: Discord interaction
        """
        try:
            if self.discord_state.get('user_session_id') != None:
                await interaction.response.send_message(
                    f"You are already logged in. Please log out first. Session ID : {self.discord_state.get('user_session_id')}",
                    ephemeral=True
                )
                return
            # Check if user is part of the server
            target_guild = self.client.get_guild(self.app_config.get_discord_target_guild_id())
            user_id = interaction.user.id
            
            if target_guild:
                try:
                    member = await target_guild.fetch_member(user_id)
                    if not member:
                        await interaction.response.send_message(
                            "You are not part of Discord Server. Access to command is restricted.",
                            ephemeral=True
                        )
                        return
                except discord.NotFound:
                    await interaction.response.send_message(
                        "You are not part of Discord Server. Access to command is restricted.",
                        ephemeral=True
                    )
                    return
            
            # If user is a member, show the login modal
            modal = LoginModal(self)
            
            await interaction.response.send_modal(modal)
            self.logger.info(f"@discord_bot.py Login modal sent to user {interaction.user.name}")
        
        except Exception as e:
            error_msg = f"Error processing login command: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            await self._send_error_response(interaction)
                             
    async def _handle_ask_command(
        self,
        interaction: discord.Interaction,
        question: str) -> None:
        """
        Handle the ask command.
        
        Args:
            interaction: Discord interaction
            question: User's question
        """
        self.logger.info(f"@discord_bot.py User {interaction.user.name} asked: {question}")
        user = interaction.user
        user_id= interaction.user.id
        request_in_dm = isinstance(interaction.channel, discord.DMChannel)
    
        target_guild = self.client.get_guild(self.app_config.get_discord_target_guild_id())
        user_has_mod_role= None
        member = None
        # Reset all states

        if target_guild:
            try:
                member = await target_guild.fetch_member(interaction.user.id)
                if member:
                    required_role_name = self.app_config.get_discord_mod_role_name() 
                    user_has_mod_role = any(
                        role.name == required_role_name for role in member.roles
                    )
                    
                else:
                    return "You are not part of AIM Discord Server. Access to command is restricted."

                    
            except discord.NotFound:
                return "You are not part of AIM Discord Server. Access to command is restricted."
        self.discord_state.update(user=user, target_guild=target_guild, request_in_dm=request_in_dm,user_id=user_id, guild_user = member, user_has_mod_role=user_has_mod_role,discord_post_channel_name = self.app_config.get_discord_post_channel_name(),  discord_mod_role_name = self.app_config.get_discord_mod_role_name())
        self.firestore.update_collection("direct_messages" if request_in_dm else "guild_messages" )
         
        try:
            if not await self._validate_channel(interaction):
                return
            if not await self._validate_question_length(interaction, question):
                return
            
            await self._process_and_respond(interaction, question)

        except Exception as e:
            error_msg = f"Error processing ask command: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            await self._send_error_response(interaction)

    async def _validate_channel(self, interaction: discord.Interaction) -> bool:
        """Validate if command is used in correct channel"""
        if not self.discord_state.get('request_in_dm') and interaction.channel.id != self.app_config.get_discord_allowed_chat_id():
            await interaction.response.send_message(
                "Please use this command in the designated channel: #general",
                ephemeral=True
            )
            return False
        return True

    async def _validate_question_length(
        self,
        interaction: discord.Interaction,
        question: str) -> bool:
        """Validate question length"""
        if len(question) > self.config.max_question_length:
            await interaction.response.send_message(
                f"Question too long ({len(question)} characters). "
                f"Please keep under {self.config.max_question_length} characters.",
                ephemeral=True
            )
            return False
        return True

    async def _process_and_respond(
        self,
        interaction: discord.Interaction,
        question: str ) -> None:
        """Process question and send response"""
        try:
            
            await interaction.response.defer(thinking=True)
            global task_message
            task_message = await interaction.edit_original_response(content="⋯ Understanding your question")
            await self.utils.start_animation(task_message)
            response = await self.agents.process_question(question)
            await self._send_chunked_response(interaction, response)
            self.logger.info(f"@discord_bot.py Successfully processed question for {interaction.user.name}")
            
            
            
            self.firestore.update_message("user_message", question)
            await self.vector_store.store_to_vector_db()
            await self.utils.perform_raptor_tree_update()
            document_id = await self.firestore.push_message()
            self.logger.info(f"@discord_bot.py Message pushed with document ID: {document_id}")

        except asyncio.TimeoutError:
            self.logger.error(f"@discord_bot.py Response generation timed out")
            await self._send_error_response(
                interaction,
                "Sorry, the response took too long to generate. Please try again."
            )
        except Exception as e:
            self.logger.error(f"@discord_bot.py Error processing question at discord class: {str(e)}", exc_info=True)
            await self._send_error_response(interaction)

    async def setup_verify_button(self):
        self.logger.info(self.app_config.get_discord_target_guild_id())
        channel = self.client.get_channel(self.app_config.get_discord_verification_id())  # Verify channel ID
        if channel:
            view = discord.ui.View(timeout=None)
            view.add_item(VerifyButton(self.app_config))
            await channel.send("Click here to verify", view=view)

    async def _send_chunked_response( self,interaction: discord.Interaction,response: str) -> None:
        """Send response in chunks if needed"""
        try:
            ground_sources = self.utils.get_ground_sources()
            # Create buttons for each URL
            buttons = []
            for url in ground_sources:
                if isinstance(url, str):
                    domain = urlparse(url).netloc
                    button = discord.ui.Button(label=domain, url=url, style=discord.ButtonStyle.link)
                    buttons.append(button)
                else:
                    self.logger.warning(f"Skipping non-string URL: {type(url)}")
            # Custom link for feedbacks
            button = discord.ui.Button(label="Feedback", url=f"https://discord.com/channels/{self.app_config.get_discord_target_guild_id()}/{self.app_config.get_discord_feedback_id()}", style=discord.ButtonStyle.primary)
            buttons.append(button)

            view = discord.ui.View()
            for button in buttons:
                view.add_item(button)

            if len(response) > self.config.max_response_length:
                chunks = [
                    response[i:i + self.config.chunk_size]
                    for i in range(0, len(response), self.config.chunk_size)
                ]
                global task_message
                await self.utils.stop_animation(task_message, chunks[0])
                for chunk in chunks[1:-1]:
                    await interaction.followup.send(content=chunk)
                await interaction.followup.send(content=chunks[-1], view=view)
            else:
                await self.utils.stop_animation(task_message, response,View=view)
            
            self.utils.clear_ground_sources()

        except Exception as e:
            self.logger.error(f"@discord_bot.py Error sending response: {str(e)}", exc_info=True)
            await self._send_error_response(interaction)

    async def _send_error_response(
        self,
        interaction: discord.Interaction,
        message: str = "Sorry, I encountered an error processing your question. Please try again.") -> None:
        """Send error response to user"""
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    content=message,
                    ephemeral=True
                )
            else:
                await interaction.followup.send(
                    content=message,
                    ephemeral=True
                )
        except Exception as e:
            self.logger.error(f"@discord_bot.py Error sending error response: {str(e)}", exc_info=True)

    async def _handle_ready(self):
        try:
            await self.tree.sync()
            self.logger.info(f'Bot is ready! Logged in as {self.client.user}')
            await self.setup_verify_button()  # Set up the verify button when the bot starts
        except Exception as e:
            self.logger.error(f"@discord_bot.py Error in ready event: {str(e)}", exc_info=True)

    async def start(self) -> None:
        """Start the Discord bot"""
        try:
            await self.client.start(self.config.token)
        except Exception as e:
            self.logger.error(f"@discord_bot.py Failed to start bot: {str(e)}", exc_info=True)
            raise

    async def close(self) -> None:
        """Close the Discord bot"""
        try:
            await self.client.close()
        except Exception as e:
            self.logger.error(f"@discord_bot.py Error closing bot: {str(e)}", exc_info=True)
