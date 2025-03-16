from utils.common_imports import *

class Superior_Agent_Tools:
    
    def __init__(self,firestore,discord_state,utils, app_config, live_status_agent, rag_search_agent,discord_agent,logger,group_chat):
        self.group_chat = group_chat

        self.conversations = {}
        self.app_config = app_config
        self.firestore = firestore
        self.discord_state = discord_state
        self.utils = utils
        self.live_status_agent = live_status_agent
        self.rag_search_agent = rag_search_agent
        self.discord_agent = discord_agent
        self.logger = logger
        self.client = genai_vertex.Client(api_key=self.app_config.get_api_key())
        self.model_id = "gemini-2.0-flash-exp"
        self.discord_client = self.discord_state.get('discord_client')
        self.guild = self.discord_state.get('target_guild')
        self.user_id=self.discord_state.get('user_id')
        self.user= self.discord_state.get('user')
        self.google_search_tool = Tool(google_search=GoogleSearch())
    
    def get_final_url(self,url):
        try:
            response = requests.get(url, allow_redirects=True)
            return response.url
        except Exception as e:
            self.logger.error(e)
            return e  

    async def access_rag_search_agent(self, instruction_to_agent: str, special_instructions: str):
        self.logger.info(f"Action Model : accessing search agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        self.group_chat.update_text(instruction_to_agent)
        
        try:
            response = await self.rag_search_agent.determine_action(instruction_to_agent,special_instructions)
            return response
        except Exception as e:
            self.logger.error(f"Error in access search agent: {str(e)}")
            return f"Search Agent Not Responsive"
         
    async def access_discord_agent(self, instruction_to_agent: str,special_instructions: str):
        self.logger.info(f"Action Model : accessing discord agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        self.group_chat.update_text(instruction_to_agent)

        try:
            response = await self.discord_agent.determine_action(instruction_to_agent,special_instructions)
            
            return response
        except Exception as e:
            self.logger.error(f"Error in access discord agent: {str(e)}")
            return f"Discord Agent Not Responsive"
        
    async def get_user_profile_details(self) -> str:
        """Retrieve user profile details from the Discord server"""
        self.guild = self.discord_state.get('target_guild')
        self.user_id = self.discord_state.get('user_id')
        self.logger.info(f"Discord Model: Handling user profile details request for user ID: {user_id}")

        if not request_in_dm:
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        try:
            # If no user_id is provided, use the requester's ID
            if not user_id:
                user_id = self.user_id

            member = await self.guild.fetch_member(user_id)
            if not member:
                return f"Unable to find user with ID {user_id} in the server."

            # Fetch user-specific data (customize based on your server's setup)
            join_date = member.joined_at.strftime("%Y-%m-%d")
            roles = [role.name for role in member.roles if role.name != "@everyone"]
            
            # You might need to implement these functions based on your server's systems
            # activity_points = await self.get_user_activity_points(user_id)
            # leaderboard_position = await self.get_user_leaderboard_position(user_id)
            # - Activity Points: {activity_points}
            # - Leaderboard Position: {leaderboard_position}

            profile_info = f"""
            User Profile for {member.name}#{member.discriminator}:
            - Join Date: {join_date}
            - Roles: {', '.join(roles)}
            - Server Nickname: {member.nick if member.nick else 'None'}
            """

            return profile_info.strip()

        except discord.errors.NotFound:
            return f"User with ID {user_id} not found in the server."
        except Exception as e:
            self.logger.error(f"Error retrieving user profile: {str(e)}")
            return f"An error occurred while retrieving the user profile: {str(e)}"
    
    async def get_discord_server_info(self) -> str:
             
        self.discord_client = self.discord_state.get('discord_client')
        self.logger.info(f"Initialized Discord Client : {self.discord_client}")
        self.guild = self.discord_state.get("target_guild")
        
        self.logger.info(f"Initialized Discord Guild : {self.guild}")
        """Create discord forum post callable by model"""

        
        self.logger.info(f"Discord Model : Handling discord server info request with context")
                
        
        return f"""1.Sparky Discord Server - Sparky Discord Server is a place where ASU Alumni's or current students join to hangout together, have fun and learn things about ASU together and quite frankly!
        2. Sparky Discord Bot -  AI Agent built to help people with their questions regarding ASU related information and sparky's discord server. THis AI Agent can also perform discord actions for users upon request."""
    
    async def access_live_status_agent(self, instruction_to_agent: str, special_instructions: str):
        self.logger.info(f"Action Model : accessing live status agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        self.group_chat.update_text(instruction_to_agent)
        
        try:
            response = await self.live_status_agent.determine_action(instruction_to_agent,special_instructions)
            return response
        except Exception as e:
            self.logger.error(f"Error in deep search agent: {str(e)}")
            return "I apologize, but I couldn't retrieve the information at this time."
              
    async def send_bot_feedback(self, feedback: str) -> str:
        self.user = self.discord_state.get('user') 
        self.discord_client = self.discord_state.get('discord_client')
        
        await self.utils.update_text("Opening feedbacks...")
        
        self.logger.info("Contact Model: Handling contact request for server feedback")

        try:
            # Find the feedbacks channel
            feedbacks_channel = discord.self.utils.get(self.discord_client.get_all_channels(), name='feedback')
            if not feedbacks_channel:
                return "feedbacks channel not found. Please ensure the channel exists."

            # Create an embed for the feedback
            embed = discord.Embed(title="New Server feedback", color=discord.Color.green())
            embed.add_field(name="feedback", value=feedback, inline=False)
            embed.set_footer(text=f"Suggested by {self.user.name}")

            # Send the feedback to the channel
            message = await feedbacks_channel.send(embed=embed)

            # Add reactions for voting
            await message.add_reaction('👍')
            await message.add_reaction('👎')
            
            self.utils.update_ground_sources([message.jump_url])
            
            return f"Your feedback has been successfully submitted."

        except Exception as e:
            self.logger.error(f"Error sending feedback: {str(e)}")
            return f"An error occurred while sending your feedback: {str(e)}"
    

    def _save_message(self, user_id: str, role: str, content: str):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({
            "role": role,
            "parts": [{"text": content}]
        })
        
        # Limit the conversation length to 3 messages per user
        if len(self.conversations[user_id]) > 3:
            self.conversations[user_id].pop(0)

    async def access_google_agent(self, original_query: str, detailed_query: str, generalized_query: str, relative_query: str, categories: list):
        self.firestore.update_message("category", categories)
        
        user_id = self.discord_state.get('user_id')
        responses=[]
        self.logger.info(f"Action Model: accessing Google Search with instruction {original_query}")
        try:
            # Perform database search
            queries = [
                {"search_bar_query": original_query},
                {"search_bar_query": detailed_query},
                {"search_bar_query": generalized_query},
                {"search_bar_query": relative_query}
            ]
            for query in queries:
                response = await self.utils.perform_database_search(query["search_bar_query"], categories) or []
                responses.append(response)

            responses = [resp for resp in responses if resp]
        except:
            self.logger.error("No results found in database")
            pass
        # Get chat history
        

        # Prepare the prompt
        prompt = f"""
        
        {self.app_config.get_google_agent_prompt()}
        
        - If applicable, you may use the related database information : {responses}
        

        User's Query: {original_query}

        Deliver a direct, actionable response that precisely matches the query's specificity."""
        
        try:     
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=GenerateContentConfig(
                    tools=[self.google_search_tool],
                    response_modalities=["TEXT"],
                    system_instruction=f"{self.app_config.get_google_agent_instruction()}",
                    max_output_tokens=600
                )
            )
            self.logger.info(f"GOogle search agent response : {response}")
            grounding_sources = [self.get_final_url(chunk.web.uri) for candidate in response.candidates if candidate.grounding_metadata and candidate.grounding_metadata.grounding_chunks for chunk in candidate.grounding_metadata.grounding_chunks if chunk.web]
            
            self.utils.update_ground_sources(grounding_sources)
            
            response_text = "".join([part.text for part in response.candidates[0].content.parts if part.text])


            # Save the interaction to chat history
            self._save_message(user_id, "user", original_query)
            self._save_message(user_id, "model", response_text)

            self.logger.info(response_text)

            if not response_text:
                self.logger.error("No response from Google Search")
                return None
            return response_text
        except Exception as e:
            self.logger.info(f"Google Search Exception {e}")
            return responses 