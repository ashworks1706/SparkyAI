
class ASUDiscordBot:
    
    """Discord bot for handling ASU-related questions"""

    def __init__(self, rag_pipeline, config: Optional[BotConfig] = None):
        """
        Initialize the Discord bot.
        
        Args:
            rag_pipeline: RAG pipeline instance
            config: Optional bot configuration
        """
        logger.info("\nInitializing ASUDiscordBot")
        self.config = config or BotConfig(app_config)
        self.rag_pipeline = rag_pipeline
        
        # Initialize Discord client
        
        self.client = discord_state.get('discord_client')
        self.tree = app_commands.CommandTree(self.client)
        self.guild = self.client.get_guild(1256076931166769152)
        self.service = Service(ChromeDriverManager().install())
        
        # Register commands and events
        self._register_commands()
        self._register_events()
   
    def _register_commands(self) -> None:
        """Register Discord commands"""
        
        @self.tree.command(
            name=self.config.command_name,
            description=self.config.command_description
        )
        async def ask(interaction: discord.Interaction, question: str):
            await self._handle_ask_command(interaction, question)
        
    def _register_events(self) -> None:
        """Register Discord events"""
        
        @self.client.event
        async def on_ready():
            await self._handle_ready()
            
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
        logger.info(f"User {interaction.user.name} asked: {question}")
        user = interaction.user
        user_id= interaction.user.id
        request_in_dm = isinstance(interaction.channel, discord.DMChannel)
        self.guild = self.client.get_guild(1256076931166769152)
        target_guild = self.client.get_guild(1256076931166769152)
        user_has_mod_role= None
        member = None
        user_voice_channel_id=None
        # Reset all states

        if target_guild:
            try:
                member = await target_guild.fetch_member(interaction.user.id)
                if member:
                    required_role_name = "mod" 
                    user_has_mod_role = any(
                        role.name == required_role_name for role in member.roles
                    )
                    
                    # Check voice state
                    if member.voice:
                        user_voice_channel_id = member.voice.channel.id
                else:
                    return "You are not part of Sparky Discord Server. Access to command is restricted."

                    
            except discord.NotFound:
                return "You are not part of Sparky Discord Server. Access to command is restricted."
        await asu_scraper.__login__(app_config.get_handshake_user(),app_config.get_handshake_pass() )
        discord_state.update(user=user, target_guild=target_guild, request_in_dm=request_in_dm,user_id=user_id, guild_user = member, user_has_mod_role=user_has_mod_role,user_voice_channel_id=user_voice_channel_id)
        firestore.update_collection("direct_messages" if request_in_dm else "guild_messages" )
         
        try:
            if not await self._validate_channel(interaction):
                return
            if not await self._validate_question_length(interaction, question):
                return
            
            await self._process_and_respond(interaction, question)

        except Exception as e:
            error_msg = f"Error processing ask command: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await self._send_error_response(interaction)

    async def _validate_channel(self, interaction: discord.Interaction) -> bool:
        """Validate if command is used in correct channel"""
        if not discord_state.get('request_in_dm') and interaction.channel.id != 1323387010886406224:
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
            await utils.start_animation(task_message)
            response = await self.rag_pipeline.process_question(question)
            await self._send_chunked_response(interaction, response)
            logger.info(f"Successfully processed question for {interaction.user.name}")
            await asu_store.store_to_vector_db()
            await (google_sheet.increment_function_call(discord_state.get('user_id'), 'G'))
            await (google_sheet.increment_function_call(discord_state.get('user_id'), 'N'))
            await (google_sheet.update_user_column(interaction.user.id, 'E', question))
            await (google_sheet.update_user_column(interaction.user.id, 'F', response))
            
            await google_sheet.perform_updates()
            
            firestore.update_message("user_message", question)
            document_id = await firestore.push_message()
            logger.info(f"Message pushed with document ID: {document_id}")

        except asyncio.TimeoutError:
            logger.error("Response generation timed out")
            await self._send_error_response(
                interaction,
                "Sorry, the response took too long to generate. Please try again."
            )
        except Exception as e:
            logger.error(f"Error processing question at discord class: {str(e)}", exc_info=True)
            await self._send_error_response(interaction)

    async def setup_verify_button(self):
        channel = self.client.get_channel(1323386003896926248)  # Verify channel ID
        if channel:
            view = discord.ui.View(timeout=None)
            view.add_item(VerifyButton())
            await channel.send("Click here to verify", view=view)

    async def _send_chunked_response( self,interaction: discord.Interaction,response: str) -> None:
        """Send response in chunks if needed"""
        try:
            ground_sources = utils.get_ground_sources()
            # Create buttons for each URL
            buttons = []
            for url in ground_sources:
                domain = urlparse(url).netloc
                button = discord.ui.Button(label=domain, url=url, style=discord.ButtonStyle.link)
                buttons.append(button)
            # Custom link for feedbacks
            button = discord.ui.Button(label="Feedback", url="https://discord.com/channels/1256076931166769152/1323386415337177150", style=discord.ButtonStyle.link)
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
                await utils.stop_animation(task_message, chunks[0])
                for chunk in chunks[1:-1]:
                    await interaction.followup.send(content=chunk)
                await interaction.followup.send(content=chunks[-1], view=view)
            else:
                await utils.stop_animation(task_message, response,View=view)
            
            utils.clear_ground_sources()

        except Exception as e:
            logger.error(f"Error sending response: {str(e)}", exc_info=True)
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
            logger.error(f"Error sending error response: {str(e)}", exc_info=True)

    async def _handle_ready(self):
        try:
            await self.tree.sync()
            logger.info(f'Bot is ready! Logged in as {self.client.user}')
            await self.setup_verify_button()  # Set up the verify button when the bot starts
        except Exception as e:
            logger.error(f"Error in ready event: {str(e)}", exc_info=True)

    async def start(self) -> None:
        """Start the Discord bot"""
        try:
            await self.client.start(self.config.token)
        except Exception as e:
            logger.error(f"Failed to start bot: {str(e)}", exc_info=True)
            raise

    async def close(self) -> None:
        """Close the Discord bot"""
        try:
            await self.client.close()
        except Exception as e:
            logger.error(f"Error closing bot: {str(e)}", exc_info=True)
