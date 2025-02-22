class ActionModel:
    
action_model = genai.GenerativeModel(
    
    model_name="gemini-1.5-flash",
    
    generation_config={
        "temperature": 0.0, 
        "top_p": 0.1,
        "top_k": 40,
        "max_output_tokens": 3100,
        "response_mime_type": "text/plain",
    },
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
    
    system_instruction = f""" {app_config.get_action_agent_instruction()}""",
    
    tools=[
        genai.protos.Tool(
            function_declarations=[
                            
                genai.protos.FunctionDeclaration(
                    name="access_search_agent",
                    description="Has ability to search for ASU-specific Targeted , real-time information extraction related to Jobs, Scholarships, Library Catalog, News, Events, Social Media, Sport Updates, Clubs",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "instruction_to_agent": content.Schema(
                                type=content.Type.STRING,
                                description="Tasks for the agent"
                            ),
                            "special_instructions": content.Schema(
                                type=content.Type.STRING,
                                description="Remarks about previous search or Special Instructions to Agent"
                            ),
                        },
                        required= ["instruction_to_agent","special_instructions"],
                    ),
                ),
                
                genai.protos.FunctionDeclaration(
                    name="access_discord_agent",
                    description="Has ability to post announcement/event/poll and connect user to moderator/helper request",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "instruction_to_agent": content.Schema(
                                type=content.Type.STRING,
                                description="Tasks for the agent"
                            ),
                            "special_instructions": content.Schema(
                                type=content.Type.STRING,
                                description="Remarks about previous search or Special Instructions to Agent"
                            ),
                        },
                        required= ["instruction_to_agent","special_instructions"]
                    ),   
                ),
                
                genai.protos.FunctionDeclaration(
                    name="send_bot_feedback",
                    description="Submits user's feedbacks about sparky",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "feedback": content.Schema(
                                type=content.Type.STRING,
                                description="Feedback by the user"
                            ),
                        },
                    required=["feedback"]
                    ),
                ),
                   
                genai.protos.FunctionDeclaration(
                    name="get_discord_server_info",
                    description="Get Sparky Discord Server related Information",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "context": content.Schema(
                                type=content.Type.STRING,
                                description="Context of Information"
                            ),
                        },
                          
                    ),
                ),
                 
                genai.protos.FunctionDeclaration(
                    name="access_live_status_agent",
                    description="Has ability to fetch realtime live shuttle/bus, library and StudyRooms status.",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "instruction_to_agent": content.Schema(
                                type=content.Type.STRING,
                                description="Tasks for Live Status Agent"
                            ),
                            "special_instructions": content.Schema(
                                type=content.Type.STRING,
                                description="Special Instructions to the agent"
                            ),
                        },
                        required= ["instruction_to_agent","special_instructions"],
                    ),
                ),
                 
                genai.protos.FunctionDeclaration(
                    name="get_user_profile_details",
                    description="Get Sparky Discord Server related Information",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "context": content.Schema(
                                type=content.Type.STRING,
                                description="Context of Information"
                            ),
                        },
                          
                    ),
                ),

                genai.protos.FunctionDeclaration(
                    name="access_google_agent",
                    description="Performs Google Search through to provide rapid result summary",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        properties={
                            "original_query": content.Schema(
                                type=content.Type.STRING,
                                description="Original Query to Search"
                            ),
                            "detailed_query": content.Schema(
                                type=content.Type.STRING,
                                description="Detailed query related to the question"
                            ),
                            "generalized_query": content.Schema(
                                type=content.Type.STRING,
                                description="General query related to the question"
                            ),
                            "relative_query": content.Schema(
                                type=content.Type.STRING,
                                description="Other query related to the question"
                            ),
                            "categories": content.Schema(
                                type=content.Type.ARRAY,
                                items=content.Schema(
                                    type=content.Type.STRING,
                                    enum=[
                                        "libraries_status", 
                                        "shuttles_status", 
                                        "clubs_info", 
                                        "scholarships_info", 
                                        "job_updates", 
                                        "library_resources",  
                                        "classes_info", 
                                        "events_info", 
                                        "news_info", 
                                        "social_media_updates", 
                                    ]
                                ),
                                description="Documents Category Filter"
                            ),

                        },
                        required=["original_query","detailed_query","generalized_query","relative_query","categories"]
                    ),
                ),    
            ],
        ),
    ],
    tool_config={'function_calling_config': 'AUTO'},
)
    def __init__(self):
        self.model = action_model
        self.chat = None
        self.functions = Action_Model_Functions()
        self.last_request_time = time.time()
        self.request_counter = 0

    def _initialize_model(self):
        if not self.model:
            raise Exception("Model is not available.")
        current_time = time.time()
        if current_time - self.last_request_time < 1.0:
            raise Exception("Rate limit exceeded. Please try again later.")
        self.last_request_time = current_time
        self.request_counter += 1
        self.chat = self.model.start_chat(enable_automatic_function_calling=True)

    async def execute_function(self, function_call: Any) -> str:
        function_mapping = {
            'access_search_agent': self.functions.access_search_agent,
            'access_google_agent': self.functions.access_google_agent,
            'access_discord_agent': self.functions.access_discord_agent,
            'send_bot_feedback': self.functions.send_bot_feedback,
            'access_live_status_agent': self.functions.access_live_status_agent,
            'get_user_profile_details': self.functions.get_user_profile_details,
            'get_discord_server_info': self.functions.get_discord_server_info,
        }

        function_name = function_call.name
        function_args = function_call.args

        if function_name not in function_mapping:
            raise ValueError(f"Unknown function: {function_name}")
        
        function_to_call = function_mapping[function_name]
        return await function_to_call(**function_args)

    async def process_gemini_response(self, response: Any) -> tuple[str, bool, Any]:
        text_response = ""
        has_function_call = False
        function_call = None
        logger.info(response)

        for part in response.parts:
            if hasattr(part, 'text') and part.text.strip():
                text_response += f"\n{part.text.strip()}"
                firestore.update_message("action_agent_message", f"Text Response : {text_response} ")
            if hasattr(part, 'function_call') and part.function_call:
                has_function_call = True
                function_call = part.function_call
                temp_func =  {
                "function_call": {
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args)
                    }
                }
                firestore.update_message("action_agent_message", json.dumps(temp_func, indent=2))

        return text_response, has_function_call, function_call

    async def determine_action(self, query: str) -> List[str]:
        try:
            final_response=""
            self._initialize_model()
            responses = []
            prompt = f"""
            ### Context:
            - Current Date and Time: {datetime.now().strftime('%H:%M %d') + ('th' if 11<=int(datetime.now().strftime('%d'))<=13 else {1:'st',2:'nd',3:'rd'}.get(int(datetime.now().strftime('%d'))%10,'th')) + datetime.now().strftime(' %B, %Y') }
            - User Query: {query}
            {app_config.get_action_agent_prompt()}
            """
            
            response = await self.chat.send_message_async(prompt)
            logger.info(f"RAW TEST RESPONSE : {response}")
            
            while True:
                text_response, has_function_call, function_call = await self.process_gemini_response(response)
                responses.append(text_response)
                final_response += text_response
                if not has_function_call:
                    break
                function_result = await self.execute_function(function_call)
                firestore.update_message("action_agent_message", f"""(User cannot see this response) System Generated - \n{function_call.name}\nResponse: {function_result}\nAnalyze the response and answer the user's question.""")
                logger.info("\nAction Model @ Function result is: %s", function_result)
                response = await self.chat.send_message_async(f"""(User cannot see this response) System Generated - \n{function_call.name}\nResponse: {function_result}\nAnalyze the response and answer the user's question.""")
                
            final_response = " ".join(response.strip() for response in responses if response.strip())
            
            return final_response.strip()
        
        except Exception as e:
            logger.error(f"Error in determine_action: {e}")
            return ["I'm sorry, I couldn't generate a response. Please try again."]