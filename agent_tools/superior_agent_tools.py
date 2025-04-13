from utils.common_imports import *

class Superior_Agent_Tools:
    
    # This class contains the tools and methods for the superior agent to interact with other agents and perform various tasks.
    def __init__(self, vector_store, asu_data_processor, firestore, discord_state, utils, app_config, shuttle_status_agent, discord_agent, courses_agent, library_agent, news_media_agent, scholarship_agent, sports_agent, student_clubs_events_agent, student_jobs_agent, logger, group_chat):
        self.group_chat = group_chat

        self.conversations = {}
        self.vector_store = vector_store
        self.asu_data_processor = asu_data_processor
        self.app_config = app_config
        self.firestore = firestore
        self.discord_state = discord_state
        self.utils = utils
        self.shuttle_status_agent = shuttle_status_agent
        self.discord_agent = discord_agent
        self.courses_agent = courses_agent
        self.library_agent = library_agent
        self.news_media_agent = news_media_agent
        self.scholarship_agent = scholarship_agent
        self.sports_agent = sports_agent
        self.student_clubs_events_agent = student_clubs_events_agent
        self.student_jobs_agent = student_jobs_agent
        self.logger = logger
        self.client = genai_vertex.Client(api_key=self.app_config.get_api_key())
        self.model_id = "gemini-2.0-flash-exp"
        self.discord_client = self.discord_state.get('discord_client')
        self.guild = self.discord_state.get('target_guild')
        self.user_id = self.discord_state.get('user_id')
        self.user = self.discord_state.get('user')
        self.google_search_tool = Tool(google_search=GoogleSearch())

    def get_final_url(self, url):
        try:
            response = requests.get(url, allow_redirects=True)
            return response.url
        except Exception as e:
            self.logger.error(e)
            return e  
         
    async def access_discord_agent(self, instruction_to_agent: str, special_instructions: str):
        self.logger.info(f"@superior_agent_tools.py Action Model : accessing discord agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        self.group_chat.update_text(instruction_to_agent)

        try:
            response = await self.discord_agent.determine_action(instruction_to_agent, special_instructions)
            return response
        except Exception as e:
            self.logger.error(f"@superior_agent_tools.py Error in access discord agent: {str(e)}")
            return "Discord Agent Not Responsive"
        
    async def get_user_profile_details(self) -> str:
        """Retrieve user profile details from the Discord server"""
        self.guild = self.discord_state.get('target_guild')
        self.user_id = self.discord_state.get('user_id')
        self.logger.info(f"@superior_agent_tools.py Discord Model: Handling user profile details request for user ID: {self.user_id}")

        if not self.request_in_dm:
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        try:
            # If no user_id is provided, use the requester's ID
            if not self.user_id:
                self.user_id = self.user_id

            member = await self.guild.fetch_member(self.user_id)
            if not member:
                return f"Unable to find user with ID {self.user_id} in the server."

            # Fetch user-specific data (customize based on your server's setup)
            join_date = member.joined_at.strftime("%Y-%m-%d")
            roles = [role.name for role in member.roles if role.name != "@everyone"]
            
            profile_info = f"""
            User Profile for {member.name}#{member.discriminator}:
            - Join Date: {join_date}
            - Roles: {', '.join(roles)}
            - Server Nickname: {member.nick if member.nick else 'None'}
            """

            return profile_info.strip()

        except discord.errors.NotFound:
            return f"User with ID {self.user_id} not found in the server."
        except Exception as e:
            self.logger.error(f"@superior_agent_tools.py Error retrieving user profile: {str(e)}")
            return f"An error occurred while retrieving the user profile: {str(e)}"
    
    async def access_shuttle_status_agent(self, instruction_to_agent: str, special_instructions: str):
        self.logger.info(f"@superior_agent_tools.py Action Model : accessing live status agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        self.group_chat.update_text(instruction_to_agent)
        
        try:
            response = await self.shuttle_status_agent.determine_action(instruction_to_agent, special_instructions)
            return response
        except Exception as e:
            self.logger.error(f"@superior_agent_tools.py Error in deep search agent: {str(e)}")
            return "I apologize, but I couldn't retrieve the information at this time."
    
    async def access_courses_agent(self, instruction_to_agent: str, special_instructions: str):
        self.logger.info(f"@superior_agent_tools.py Action Model : accessing courses agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        self.group_chat.update_text(instruction_to_agent)
        
        try:
            response = await self.courses_agent.determine_action(instruction_to_agent, special_instructions)
            return response
        except Exception as e:
            self.logger.error(f"@superior_agent_tools.py Error in access courses agent: {str(e)}")
            return "Courses Agent Not Responsive"
    
    async def access_student_clubs_events_agent(self, instruction_to_agent: str, special_instructions: str):
        self.logger.info(f"@superior_agent_tools.py Action Model : accessing studentclubsevents agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        self.group_chat.update_text(instruction_to_agent)
        
        try:
            response = await self.student_clubs_events_agent.determine_action(instruction_to_agent, special_instructions)
            return response
        except Exception as e:
            self.logger.error(f"@superior_agent_tools.py Error in access studentclubsevents agent: {str(e)}")
            return "Studentclubsevents Agent Not Responsive"
    
    async def access_library_agent(self, instruction_to_agent: str, special_instructions: str):
        self.logger.info(f"@superior_agent_tools.py Action Model : accessing library agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        self.group_chat.update_text(instruction_to_agent)
        
        try:
            response = await self.library_agent.determine_action(instruction_to_agent, special_instructions)
            return response
        except Exception as e:
            self.logger.error(f"@superior_agent_tools.py Error in access library agent: {str(e)}")
            return "Library Agent Not Responsive"
    
    async def access_news_media_agent(self, instruction_to_agent: str, special_instructions: str):
        self.logger.info(f"@superior_agent_tools.py Action Model : accessing news media agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        self.group_chat.update_text(instruction_to_agent)
        
        try:
            response = await self.news_agent.determine_action(instruction_to_agent, special_instructions)
            return response
        except Exception as e:
            self.logger.error(f"@superior_agent_tools.py Error in access news media agent: {str(e)}")
            return "News Agent Not Responsive"
    
    async def access_scholarship_agent(self, instruction_to_agent: str, special_instructions: str):
        self.logger.info(f"@superior_agent_tools.py Action Model : accessing scholarship agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        self.group_chat.update_text(instruction_to_agent)
        
        try:
            response = await self.scholarship_agent.determine_action(instruction_to_agent, special_instructions)
            return response
        except Exception as e:
            self.logger.error(f"@superior_agent_tools.py Error in access scholarship agent: {str(e)}")
            return "Scholarship Agent Not Responsive"
    
    async def access_sports_agent(self, instruction_to_agent: str, special_instructions: str):
        self.logger.info(f"@superior_agent_tools.py Action Model : accessing sports agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        self.group_chat.update_text(instruction_to_agent)
        
        try:
            response = await self.sports_agent.determine_action(instruction_to_agent, special_instructions)
            return response
        except Exception as e:
            self.logger.error(f"@superior_agent_tools.py Error in access sports agent: {str(e)}")
            return "Sports Agent Not Responsive"
   
    async def access_student_jobs_agent(self, instruction_to_agent: str, special_instructions: str):
        self.logger.info(f"@superior_agent_tools.py Action Model : accessing student jobs agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        self.group_chat.update_text(instruction_to_agent)
        
        try:
            response = await self.student_jobs_agent.determine_action(instruction_to_agent, special_instructions)
            return response
        except Exception as e:
            self.logger.error(f"@superior_agent_tools.py Error in access student jobs agent: {str(e)}")
            return "Student Jobs Agent Not Responsive"
    
    async def send_bot_feedback(self, feedback: str) -> str:
        self.user = self.discord_state.get('user') 
        self.discord_client = self.discord_state.get('discord_client')
        self.guild = self.discord_state.get('target_guild')
        
        await self.utils.update_text("Opening feedbacks...")
        
        self.logger.info("@superior_agent_tools.py Contact Model: Handling contact request for server feedback")

        try:
            # Find the feedbacks channel
            feedbacks_channel = discord.utils.get(self.guild.channels, name='feedbacks')
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
            
            return "Your feedback has been successfully submitted."

        except Exception as e:
            self.logger.error(f"@superior_agent_tools.py Error sending feedback: {str(e)}")
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

    async def access_rag_search_agent(self, original_query: str, detailed_query: str, generalized_query: str, relative_query: str, categories: list):
        self.firestore.update_message("category", categories)
        
        user_id = self.discord_state.get('user_id')
        responses = []
        self.logger.info(f"@superior_agent_tools.py Action Model: accessing Google Search with instruction {original_query}")
        try:
            # Perform database search
            queries = [
                {"search_bar_query": original_query},
                {"search_bar_query": detailed_query},
                {"search_bar_query": generalized_query},
                {"search_bar_query": relative_query}
            ]
            self.logger.info(f"@superior_agent_tools.py Action Model: Performing database search with queries {queries}")
            for query in queries:
                self.logger.info(f"@superior_agent_tools.py Action Model: Performing database search with query {query['search_bar_query']}")
                response = await self.utils.perform_database_search(query["search_bar_query"], categories) or []
                if response == "No documents found in database":
                    self.logger.info(f"@superior_agent_tools.py Action Model: No documents found in database for query {query['search_bar_query']}")
                    return response
                responses.append(response)
                self.logger.info(f"@superior_agent_tools.py Action Model: Database search response: {response}")
                

            responses = [resp for resp in responses if resp ]
            if not responses:
                self.logger.error("@superior_agent_tools.py No results found in database")
        except Exception as e:
            self.logger.error("@superior_agent_tools.py Error in database search ")
            self.logger.error(f"@superior_agent_tools.py Error in database search responses: {responses}")
            raise Exception(f"Error in database search {e}")

        # Prepare the prompt
        prompt = f"""
        
        {self.app_config.get_google_agent_prompt()}
        
        - Existing RAG Databse Knowledge: {responses}
        
        - Current Date and Time: {datetime.now().strftime('%H:%M %d') + ('th' if 11<=int(datetime.now().strftime('%d'))<=13 else {1:'st',2:'nd',3:'rd'}.get(int(datetime.now().strftime('%d'))%10,'th')) + datetime.now().strftime(' %B, %Y') }
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
            self.logger.info(f"@superior_agent_tools.py Google search agent response : {response}")
            
            # Safely extract grounding sources
            grounding_sources = []
            try:
                for candidate in response.candidates:
                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        for chunk in candidate.grounding_metadata.grounding_chunks:
                            if hasattr(chunk, 'web') and chunk.web and hasattr(chunk.web, 'uri'):
                                final_url = self.get_final_url(chunk.web.uri)
                                if final_url and isinstance(final_url, str):
                                    grounding_sources.append(final_url)
            except Exception as e:
                self.logger.error(f"@superior_agent_tools.py Error extracting grounding sources: {e}")
                
                self.utils.update_ground_sources(grounding_sources)
                
                # Safely extract response text
                response_text = ""
            try:
                if response.candidates and response.candidates[0].content and hasattr(response.candidates[0].content, 'parts'):
                    response_text = "".join([part.text for part in response.candidates[0].content.parts if hasattr(part, 'text') and part.text])
            except Exception as e:
                self.logger.error(f"@superior_agent_tools.py Error extracting response text: {e}")
            
            if not response_text and responses:
                response_text = str(responses)

            # Save the interaction to chat history
            user_id = self.discord_state.get('user_id')
            self._save_message(user_id, "user", original_query)
            self._save_message(user_id, "model", response_text)

            self.logger.info(response_text)
            
            if response_text and 'search agent' not in response_text.lower():
                processed_docs = await self.asu_data_processor.process_documents(
                    documents=[{
                        'content': response_text,
                        'metadata': {
                            'url': grounding_sources,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        }], 
                    search_context= self.group_chat.get_text(),
                    title = detailed_query, category = categories[0]
                )

                self.vector_store.queue_documents(processed_docs)
                self.utils.raptor_retriever.queue_raptor_tree(processed_docs)

            if not response_text:
                self.logger.error("@superior_agent_tools.py No response from Google Search")
                return None
            return response_text
        except Exception as e:
            self.logger.info(f"@superior_agent_tools.py Google Search Exception {e}")
            return responses