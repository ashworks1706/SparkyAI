from utils.common_imports import *

class Discord_Agent_Tools:
    def __init__(self,middleware,utils,logger):
        self.utils = utils
        self.middleware=middleware
        self.discord_client = middleware.get('discord_client')
        self.logger= logger
        self.logger.info(f"@discord_agent_tools.py Initialized Discord Client : {self.discord_client}")
        self.guild = self.middleware.get('target_guild')
        self.user_id=self.middleware.get('user_id')
        self.user=self.middleware.get('user')
        self.logger.info(f"@discord_agent_tools.py Initialized Discord Guild : {self.guild}")
     
    async def notify_discord_helpers(self, short_message_to_helper: str) -> str:
        self.guild = self.middleware.get('target_guild')
        self.user_id=self.middleware.get('user_id')
        self.user=self.middleware.get('user')
        self.logger.info(f"@discord_agent_tools.py Initialized Discord Guild : {self.guild}")

        if not self.request_in_dm:
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        await self.utils.update_text("Checking available discord helpers...")

        self.logger.info("@discord_agent_tools.py Contact Model: Handling contact request for helper notification")

        try:

            if not self.guild:
                return "Unable to find the server. Please try again later."

            # Check if user is already connected to a helper
            existing_channel = discord.utils.get(self.guild.channels, name=f"help-{self.user_id}")
            if existing_channel:
                self.utils.update_ground_sources([existing_channel.jump_url])
                return f"User already has an open help channel."

            # Find helpers
            helper_role = discord.utils.get(self.guild.roles, name="Helper")
            if not helper_role:
                return "Unable to find helpers. Please contact an administrator."

            helpers = [member for member in self.guild.members if helper_role in member.roles and member.status != discord.Status.offline]
            if not helpers:
                return "No helpers are currently available. Please try again later."

            # Randomly select a helper
            selected_helper = random.choice(helpers)

            # Create a private channel
            overwrites = {
                self.guild.default_role: discord.PermissionOverwrite(read_messages=False),
                self.user: discord.PermissionOverwrite(read_messages=True, send_messages=True),
                selected_helper: discord.PermissionOverwrite(read_messages=True, send_messages=True)
            }
            
            category = discord.utils.get(self.guild.categories, name="Customer Service")
            if not category:
                return "Unable to find the Customer Service category. Please contact an administrator."

            channel = await self.guild.create_text_channel(f"help-{self.user_id}", category=category, overwrites=overwrites)

            # Send messages
            await channel.send(f"{self.user.mention} and {selected_helper.mention}, this is your help channel.")
            await channel.send(f"User's message: {short_message_to_helper}")

            # Notify the helper via DM
            await selected_helper.send(f"You've been assigned to a new help request. Please check {channel.mention}")
            self.utils.update_ground_sources([channel.jump_url])
            return f"Server Helper Assigned: {selected_helper.name}\n"

        except Exception as e:
            self.logger.error(f"@discord_agent_tools.py Error notifying helpers: {str(e)}")
            return f"An error occurred while notifying helpers: {str(e)}"

    async def notify_moderators(self, short_message_to_moderator: str) -> str:
        self.guild = self.middleware.get('target_guild')
        self.user_id=self.middleware.get('user_id')
        self.user=self.middleware.get('user')
        
        self.logger.info(f"@discord_agent_tools.py Initialized Discord Guild : {self.guild}")


        if not self.middleware.get('request_in_dm'):
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        await self.utils.update_text("Checking available discord moderators...")

        self.logger.info("@discord_agent_tools.py Contact Model: Handling contact request for moderator notification")

        try:
            if not self.guild:
                return "Unable to find the server. Please try again later."

            # Check if user is already connected to a helper
            existing_channel = discord.utils.get(self.guild.channels, name=f"support-{self.user_id}")
            if existing_channel:
                self.utils.update_ground_sources([existing_channel.jump_url])
                return f"User already has an open support channel."
            # Find helpers/moderators
            self.logger.info(f"@discord_agent_tools.py Searching for users wtih {self.middleware.get("discord_mod_role_name")} role")
            self.logger.info(f"@discord_agent_tools.py from {self.guild.roles}")
            helper_role = discord.utils.get(self.guild.roles, name=self.middleware.get("discord_mod_role_name"))
            if not helper_role:
                return "Unable to find moderators. Please contact an administrator."

            helpers = [member for member in self.guild.members if helper_role in member.roles]
            if not helpers:
                return "No helpers are currently available. Please try again later."

            # Randomly select a helper
            selected_helper = random.choice(helpers)

            # Create a private channel
            overwrites = {
                self.guild.default_role: discord.PermissionOverwrite(read_messages=False),
                self.user: discord.PermissionOverwrite(read_messages=True, send_messages=True),
                selected_helper: discord.PermissionOverwrite(read_messages=True, send_messages=True)
            }
            
            category = discord.utils.get(self.guild.categories, name="Customer Service")
            if not category:
                return "Unable to find the Customer Service category. Please contact an administrator."

            channel = await self.guild.create_text_channel(f"support-{self.user_id}", category=category, overwrites=overwrites)

            # Send messages
            await channel.send(f"{self.user.mention} and {selected_helper.mention}, this is your support channel.")
            await channel.send(f"User's message: {short_message_to_moderator}")

            # Notify the helper via DM
            await selected_helper.send(f"You've been assigned to a new support request. Please check {channel.mention}")
            self.utils.update_ground_sources([channel.jump_url])
            return f"Moderator Assigned: {selected_helper.name}"

        except Exception as e:
            self.logger.error(f"@discord_agent_tools.py Error notifying moderators: {str(e)}")
            return f"An error occurred while notifying moderators: {str(e)}"

    # async def start_recording_discord_call(self,channel_id:Any) -> str: 

        
    #     self.logger.info(f"@discord_agent_tools.py Initialized Discord Guild : {self.guild}")
    #     await self.utils.update_text("Checking user permissions...")
       
    #     if not self.middleware.get('user_has_mod_role'):
    #         return "User does not have enough permissions to start recording a call. This command is only accessible by moderators. Exiting command..."

    #     if not self.middleware.get('request_in_dm'):
    #         return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

    #     if not self.middleware.get('user_voice_channel_id'):
    #         return "User is not in a voice channel. User needs to be in a voice channel to start recording. Exiting command..."

    #     self.logger.info("@discord_agent_tools.py Discord Model: Handling recording request")

    #     return f"Recording started!"

    async def create_discord_forum_post(self, title: str, category: str, body_content_1: str, body_content_2: str, body_content_3: str, link:str=None) -> str:
        self.guild = self.middleware.get('target_guild')
        
        self.logger.info(f"@discord_agent_tools.py Initialized Discord Guild : {self.guild}")
        await self.utils.update_text("Checking user permissions...")


        if not self.middleware.get('request_in_dm'):
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        self.logger.info("@discord_agent_tools.py Discord Model: Handling discord forum request with context")

        try:
            if not self.guild:
                return "Unable to find the server. Please try again later."
            try:
                
                # Find the forum channel 
                forum_channel = discord.utils.get(self.guild.forums, name=self.middleware("discord_post_channel_name"))  # Replace with your forum channel name
            except Exception as e:
                self.logger.error(f"@discord_agent_tools.py Error finding forum channel: {str(e)}")
                return f"An error occurred while finding the forum channel: {str(e)}"
            if not forum_channel:
                return "Forum channel not found. Please ensure the forum exists."

            # Create the forum post
            content = f"{body_content_1}\n\n{body_content_2}\n\n{body_content_3}".strip()
            if link:
                content+=f"\n[Link]({link})"
            try:
                self.logger.info(f"@discord_agent_tools.py Forum channel ID: {forum_channel.id if forum_channel else 'None'}")
                
                thread = await forum_channel.create_thread(
                    name=title,
                    content=content,
                )

            except Exception as e:
                
                self.logger.error(f"@discord_agent_tools.py Error creating forum thread: {str(e)}")
                return f"An error occurred while creating the forum thread: {str(e)}"
            self.logger.info(f"@discord_agent_tools.py Created forum thread {thread.message.id} {type(thread)}")
            
            self.utils.update_ground_sources([f"https://discord.com/channels/{self.guild}/{thread.id}"])
            return f"Forum post created successfully.\nTitle: {title}\nDescription: {content[:100]}...\n"
        

        except discord.errors.Forbidden:
            return "The bot doesn't have permission to create forum posts. Please contact an administrator."
        except discord.errors.HTTPException as e:
            self.logger.error(f"@discord_agent_tools.py HTTP error creating forum post: {str(e)}")
            return f"An error occurred while creating the forum post: {str(e)}"
        except Exception as e:
            self.logger.error(f"@discord_agent_tools.py Error creating forum post: {str(e)}")
            return f"An unexpected error occurred while creating the forum post: {str(e)}"
    
    async def create_discord_announcement(self, ping: str, title: str, category: str, body_content_1: str, body_content_2: str, body_content_3: str, link:str = None) -> str:
        self.discord_client = self.middleware.get('discord_client')
        self.guild = self.middleware.get('target_guild')
        
        await self.utils.update_text("Checking user permissions...")


        self.logger.info(f"@discord_agent_tools.py Discord Model: Handling discord announcement request with context")

        if not self.middleware.get('request_in_dm'):
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        if not self.middleware.get('user_has_mod_role'):
            return "User does not have enough permissions to create an announcement. This command is only accessible by moderators. Exiting command..."

        try:
            # Find the announcements channel
            announcements_channel = discord.utils.get(self.discord_client.get_all_channels(), name='announcements')
            if not announcements_channel:
                return "Announcements channel not found. Please ensure the channel exists."

            # Create the embed
            embed = discord.Embed(title=title, color=discord.Color.blue())
            embed.add_field(name="Category", value=category, inline=False)
            embed.add_field(name="Details", value=body_content_1, inline=False)
            if body_content_2:
                embed.add_field(name="Additional Information", value=body_content_2, inline=False)
            if body_content_3:
                embed.add_field(name="More Details", value=body_content_3, inline=False)
            if link:
                embed.add_field(name="Links", value=link, inline=False)

            # Send the announcement
            message = await announcements_channel.send(content="@som", embed=embed)
            self.utils.update_ground_sources([message.jump_url])
            return f"Announcement created successfully."

        except Exception as e:
            self.logger.error(f"@discord_agent_tools.py Error creating announcement: {str(e)}")
            return f"An error occurred while creating the announcement: {str(e)}"
  
    async def create_discord_event(self, title: str, time_start: str, time_end: str, description: str, img_provided:Any = None) -> str:
        self.guild = self.middleware.get('target_guild')
        
        self.logger.info(f"@discord_agent_tools.py Initialized Discord Guild : {self.guild}")
        
        await self.utils.update_text("Checking user permissions...")


        if not self.middleware.get('request_in_dm'):
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        if not self.middleware.get('user_has_mod_role'):
            return "User does not have enough permissions to create an event. This command is only accessible by moderators. Exiting command..."

        self.logger.info("@discord_agent_tools.py Discord Model: Handling discord event creation request")

        try:
            if self.guild:
                return "Unable to find the server. Please try again later."

            # Parse start and end times
            start_time = datetime.fromisoformat(time_start)
            end_time = datetime.fromisoformat(time_end)

            # Create the event
            event = await self.guild.create_scheduled_event(
                name=title,
                description=description,
                start_time=start_time,
                end_time=end_time,
                location="Discord",  # or specify a different location if needed
                privacy_level=discord.PrivacyLevel.guild_only
            )

            # If an image was provided, set it as the event cover
            if img_provided:
                await event.edit(image=img_provided)

            # Create an embed for the event announcement
            embed = discord.Embed(title=title, description=description, color=discord.Color.blue())
            embed.add_field(name="Start Time", value=start_time.strftime("%Y-%m-%d %H:%M:%S"), inline=True)
            embed.add_field(name="End Time", value=end_time.strftime("%Y-%m-%d %H:%M:%S"), inline=True)
            embed.add_field(name="Location", value="Discord", inline=False)
            embed.set_footer(text=f"Event ID: {event.id}")

            # Send the announcement to the announcements channel
            announcements_channel = discord.utils.get(self.guild.text_channels, name="announcements")
            if announcements_channel:
                await announcements_channel.send(embed=embed)
            
            self.utils.update_ground_sources([event.url])

            return f"Event created successfully.\nTitle: {title}\nDescription: {description[:100]}...\nStart Time: {start_time}\nEnd Time: {end_time}\n"

        except discord.errors.Forbidden:
            return "The bot doesn't have permission to create events. Please contact an administrator."
        except ValueError as e:
            return f"Invalid date format: {str(e)}"
        except Exception as e:
            self.logger.error(f"@discord_agent_tools.py Error creating event: {str(e)}")
            return f"An unexpected error occurred while creating the event: {str(e)}"
    
    async def search_discord(self,query:str):
        results = await self.utils.perform_web_search(optional_query=query,doc_title =query)
        return results
    
    async def create_discord_poll(self, question: str, options: List[str], channel_name: str) -> str:
        self.guild = self.middleware.get('target_guild')
        

        await self.utils.update_text("Checking user permissions...")

        if not self.middleware.get('request_in_dm'):
            return "User can only access this command in private messages. Exiting command."

        if not self.middleware.get('user_has_mod_role'):
            return "User does not have enough permissions to create a poll. This command is only accessible by moderators. Exiting command..."

        self.logger.info("@discord_agent_tools.py Discord Model: Handling discord poll creation request")

        try:
            if not self.guild:
                return "Unable to find the server. Please try again later."

            # Find the specified channel
            channel = discord.utils.get(self.guild.text_channels, name=channel_name)
            if not channel:
                return f"Channel '{channel_name}' not found. Please check the channel name and try again."

            # Create the poll message
            poll_message = f"📊 **{question}**\n\n"
            emoji_options = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
            try:
                for i, option in enumerate(options):  # Limit to 10 options
                    poll_message += f"{emoji_options[i]} {option}\n"
                    
            except Exception as e:
                self.logger.error(f"@discord_agent_tools.py Error creating poll options: {str(e)}")
                return f"An unexpected error occurred while creating poll options: {str(e)}"
            
            # Send the poll message
            try:
                poll = await channel.send(poll_message)
            except Exception as e:
                self.logger.error(f"@discord_agent_tools.py Error sending poll message: {str(e)}")
                return  f"An unexpected error occurred while sending poll: {str(e)}"
            
            self.utils.update_ground_sources([poll.jump_url])  

            # Add reactions
            try:
                
                for i in range(len(options)):
                    await poll.add_reaction(emoji_options[i])
            except Exception as e:
                self.logger.error(f"@discord_agent_tools.py Error adding reactions to poll: {str(e)}")
                return f"An unexpected error occurred while adding reactions to poll: {str(e)}"
            
            return f"Poll created successfully in channel '{channel_name}'.\nQuestion: {question}\nOptions: {', '.join(options)}"

        except discord.errors.Forbidden:
            return "The bot doesn't have permission to create polls or send messages in the specified channel. Please contact an administrator."
        except Exception as e:
            
            self.logger.error(f"@discord_agent_tools.py Error creating poll: {str(e)}")
            return f"An unexpected error occurred while creating the poll: {str(e)}"