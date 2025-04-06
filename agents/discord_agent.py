from utils.common_imports import *
class DiscordModel:
    
    def __init__(self, firestore,genai,app_config,logger, discord_agent_tools, discord_state):
        self.discord_state = discord_state
        self.logger= logger
        self.app_config = app_config
        self.firestore = firestore
        self.agent_tools = discord_agent_tools

        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.0, 
                "top_p": 0.1,
                "top_k": 40,
                "max_output_tokens": 2500,
                "response_mime_type": "text/plain",
            },safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            },

        system_instruction = f""" {app_config.get_discord_agent_instruction()}
        """,
            tools=[
                genai.protos.Tool(
                    function_declarations=[
                            
                        genai.protos.FunctionDeclaration(
                            name="notify_moderators",
                            description="Contacts Discord moderators (Allowed only in Private Channels)",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "short_message_to_moderator": content.Schema(
                                        type=content.Type.STRING,
                                        description="Message for moderators "
                                    ),
                                },
                            ),
                        ),
                        genai.protos.FunctionDeclaration(
                            name="notify_discord_helpers",
                            description="Contacts Discord helpers (Allowed only in Private Channels)",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "short_message_to_helpers": content.Schema(
                                        type=content.Type.STRING,
                                        description="Message for helpers "
                                    ),
                                },
                            ),
                        ),
                        genai.protos.FunctionDeclaration(
                            name="search_discord",
                            description="Search for messages on discord server",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "query": content.Schema(
                                        type=content.Type.STRING,
                                        description="Keywords to search"
                                    ),
                                },
                            ),
                        ),
                        
                        genai.protos.FunctionDeclaration(
                            name="create_discord_poll",
                            description="Creates a poll in a specified Discord channel (Allowed only in Private Channels)",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "question": content.Schema(
                                        type=content.Type.STRING,
                                        description="The main question for the poll"
                                    ),
                                    "options": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(type=content.Type.STRING),
                                        description="List of options for the poll (maximum 10)"
                                    ),
                                    "channel_name": content.Schema(
                                        type=content.Type.STRING,
                                        description="The name of the channel where the poll should be posted"
                                    )
                                },
                                required=["question", "options", "channel_name"]
                            ),
                        ),


                        genai.protos.FunctionDeclaration(
                            name="get_user_profile_details",
                            description="Retrieves user profile information",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "context": content.Schema(
                                        type=content.Type.STRING,
                                        description="User context"
                                    ),
                                },
                            ),
                        ),
                        genai.protos.FunctionDeclaration(
                            name="create_discord_announcement",
                            description="Creates a server announcement (Allowed to special roles only in Private Channels)",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "ping": content.Schema(
                                        type=content.Type.STRING,
                                        description="The role or user to ping with the announcement (e.g., @everyone, @role, or user ID)"
                                    ),
                                    "title": content.Schema(
                                        type=content.Type.STRING,
                                        description="The title of the announcement"
                                    ),
                                    "category": content.Schema(
                                        type=content.Type.STRING,
                                        description="The category of the announcement"
                                    ),
                                    "body_content_1": content.Schema(
                                        type=content.Type.STRING,
                                        description="The main content of the announcement"
                                    ),
                                    "body_content_2": content.Schema(
                                        type=content.Type.STRING,
                                        description="Additional content for the announcement (optional)"
                                    ),
                                    "body_content_3": content.Schema(
                                        type=content.Type.STRING,
                                        description="More details for the announcement (optional)"
                                    ),
                                    "link": content.Schema(
                                        type=content.Type.STRING,
                                        description="Links"
                                    ),
                                    
                                },
                                required=["title", "category", "body_content_1","body_content_2","body_content_3"]
                            ),
                        ),

                        genai.protos.FunctionDeclaration(
                            name="create_discord_forum_post",
                            description="Creates a new forum post (Allowed only in Private Channels)",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "title": content.Schema(
                                        type=content.Type.STRING,
                                        description="The title of the forum post"
                                    ),
                                    "category": content.Schema(
                                        type=content.Type.STRING,
                                        description="The category tag for the forum post"
                                    ),
                                    "body_content_1": content.Schema(
                                        type=content.Type.STRING,
                                        description="The main content of the forum post"
                                    ),
                                    "body_content_2": content.Schema(
                                        type=content.Type.STRING,
                                        description="Additional content for the forum post (optional)"
                                    ),
                                    "body_content_3": content.Schema(
                                        type=content.Type.STRING,
                                        description="More details for the forum post (optional)"
                                    ),
                                    "link": content.Schema(
                                        type=content.Type.STRING,
                                        description="Links"
                                    ),
                                },
                                required=["title", "category", "body_content_1","body_content_2","body_content_3"]
                            ),
                        ),
                        genai.protos.FunctionDeclaration(
                            name="create_discord_event",
                            description="Creates a new Discord event (Allowed only in Private Channels for users with required permissions)",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "title": content.Schema(
                                        type=content.Type.STRING,
                                        description="The title of the Discord event"
                                    ),
                                    "time_start": content.Schema(
                                        type=content.Type.STRING,
                                        description="The start time of the event in ISO format (e.g., '2023-12-31T23:59:59')"
                                    ),
                                    "time_end": content.Schema(
                                        type=content.Type.STRING,
                                        description="The end time of the event in ISO format (e.g., '2024-01-01T01:00:00')"
                                    ),
                                    "description": content.Schema(
                                        type=content.Type.STRING,
                                        description="The description of the event"
                                    ),
                                    "img_provided": content.Schema(
                                        type=content.Type.STRING,
                                        description="URL or file path of an image to be used as the event cover (optional)"
                                    ),
                                },
                                required=["title", "time_start", "time_end", "description"]
                            ),
                        ),
                        
                    
                    ],
                ),
            ],
            tool_config={'function_calling_config': 'ANY'},
        )
        self.chat = None
        
        self.last_request_time = time.time()
        self.request_counter = 0
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        
    def _initialize_model(self):
        if not self.model:
            return self.logger.error("@discord_agent.py Model not initialized at ActionFunction")
            
        # Rate limiting check
        current_time = time.time()
        if current_time - self.last_request_time < 1.0:  # 1 second cooldown
            raise Exception("Rate limit exceeded")
            
        self.last_request_time = current_time
        self.request_counter += 1
        user_id = self.discord_state.get("user_id")
        self.chat = self.model.start_chat(history=[],enable_automatic_function_calling=True)
        

        
    async def execute_function(self, function_call):
        """Execute the called function and return its result"""
        function_name = function_call.name
        function_args = function_call.args
        
        function_mapping = {
            'notify_moderators': self.agent_tools.notify_moderators,
            'notify_discord_helpers': self.agent_tools.notify_discord_helpers,
            'create_discord_forum_post': self.agent_tools.create_discord_forum_post,
            'create_discord_announcement': self.agent_tools.create_discord_announcement,
            'create_discord_poll': self.agent_tools.create_discord_poll,
            'search_discord': self.agent_tools.search_discord,
        }
        
        if function_name in function_mapping:
            function_to_call = function_mapping[function_name]
            func_response = await function_to_call(**function_args)
            # response = await self.chat.send_message_async(f"{function_name} response : {func_response}")
            
            if func_response:
                # self._save_message(user_id, "model", f"""(Only Visible to You) System Tools - Discord Agent Response: {func_response}""")
                return func_response
            else:
                self.logger.error(f"@discord_agent.py Error extracting text from response: {e}")
                return "Error processing response"
                
                
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
                
    
    async def determine_action(self, instruction_to_agent: str,special_instructions:str) -> str:
        """Determines and executes the appropriate action based on the user query"""
        try:
            self._initialize_model()
            final_response = ""
            # Simplified prompt that doesn't encourage analysis verbosity
            prompt = f"""
            ### Context:
            - Current Date and Time: {datetime.now().strftime('%H:%M %d') + ('th' if 11<=int(datetime.now().strftime('%d'))<=13 else {1:'st',2:'nd',3:'rd'}.get(int(datetime.now().strftime('%d'))%10,'th')) + datetime.now().strftime(' %B, %Y') }
            - Superior Agent Instruction: {instruction_to_agent}
            - Superior Agent Remarks (if any): {special_instructions}
            {self.app_config.get_discord_agent_prompt()}
            """
            
            response = await self.chat.send_message_async(prompt)
            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call: 
                    # Execute function and store only its result
                    final_response = await self.execute_function(part.function_call)
                    self.firestore.update_message("discord_agent_message", f"Function called {part.function_call}  Function Response {final_response} ")
                elif hasattr(part, 'text') and part.text.strip():
                    # Only store actual response content, skip analysis messages
                    text = part.text.strip()
                    self.firestore.update_message("discord_agent_message", f"Text Response {text} ")
                    if not text.startswith("This query") and not "can be answered directly" in text:
                        final_response = text.strip()
            
        
        # Return only the final message
            return final_response if final_response else "Hi! How can I help you with ASU or the Discord server today?"
            
        except Exception as e:
            self.logger.error(f"@discord_agent.py Discord Model : Error in determine_action: {str(e)}")
            return "I apologize, but I couldn't generate a response at this time. Please try again."
        