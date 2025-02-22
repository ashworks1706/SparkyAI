from utils.common_imports import *
class SuperiorModel:
    
    def __init__(self, firestore,genai,app_config):
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash",
    
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
            
            system_instruction = f""" {self.app_config.get_superior_agent_instruction()}""",
            
            tools=[
                genai.protos.Tool(
                    function_declarations=[
                                    
                        genai.protos.FunctionDeclaration(
                            name="access_rag_search_agent",
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
        self.chat = None
        self.firestore = firestore
        self.app_config = app_config
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


    async def process_gemini_response(self, response: Any) -> tuple[str, bool, Any]:
        text_response = ""
        has_function_call = False
        function_call = None
        logger.info(response)

        for part in response.parts:
            if hasattr(part, 'text') and part.text.strip():
                text_response += f"\n{part.text.strip()}"
                self.firestore.update_message("superior_agent_message", f"Text Response : {text_response} ")
            if hasattr(part, 'function_call') and part.function_call:
                has_function_call = True
                function_call = part.function_call
                temp_func =  {
                "function_call": {
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args)
                    }
                }
                self.firestore.update_message("superior_agent_message", json.dumps(temp_func, indent=2))

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
            {self.app_config.get_superior_agent_prompt()}
            """
            
            response = await self.chat.send_message_async(prompt)
            logger.info(f"RAW TEST RESPONSE : {response}")
            
            while True:
                text_response, has_function_call, function_call = await self.process_gemini_response(response)
                responses.append(text_response)
                final_response += text_response
                if not has_function_call:
                    break
                function_result = await super().execute_function(function_call)
                self.firestore.update_message("superior_agent_message", f"""(User cannot see this response) System Generated - \n{function_call.name}\nResponse: {function_result}\nAnalyze the response and answer the user's question.""")
                logger.info("\nAction Model @ Function result is: %s", function_result)
                response = await self.chat.send_message_async(f"""(User cannot see this response) System Generated - \n{function_call.name}\nResponse: {function_result}\nAnalyze the response and answer the user's question.""")
                
            final_response = " ".join(response.strip() for response in responses if response.strip())
            
            return final_response.strip()
        
        except Exception as e:
            logger.error(f"Error in determine_action: {e}")
            return ["I'm sorry, I couldn't generate a response. Please try again."]