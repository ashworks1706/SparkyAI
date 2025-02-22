from utils.common_imports import *
class Live_Status_Model:
    
    def __init__(self,firestore,genai,app_config,logger):
        self.logger = logger
        self.app_config= app_config
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.0,
                "top_p": 0.1,
                "top_k": 40,
                "max_output_tokens": 2500,
                "response_mime_type": "text/plain",
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            },
            system_instruction = f"""
            {self.app_config.get_live_status_agent_instruction}
            """,
            tools=[
                genai.protos.Tool(
                    function_declarations=[

                        genai.protos.FunctionDeclaration(
                            name="get_live_library_status",
                            description="Retrieves Latest Information regarding ASU Library Status",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={

                                    "status_type": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                            enum=[
                                                "Availability", 
                                                "StudyRoomsAvailability", 
                                            ]
                                        ),
                                        description="Checks if library is open or close and study rooms availability"
                                    ),
                                    "library_names": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                            enum=[
                                        "Tempe Campus - Noble Library",
                                        "Tempe Campus - Hayden Library",
                                        "Downtown Phoenix Campus - Fletcher Library",
                                        "West Campus - Library",
                                        "Polytechnic Campus - Library",      
                                        ]
                                        ),
                                        description="Library Name"
                                    ),
                                    "date": content.Schema(
                                        type=content.Type.STRING,
                                        description="[ Month Prefix + Date ] (ex. DEC 09, JAN 01, FEB 21, MAR 23)",
                                    ),
                                },
                                    required=["status_type","library_names","date"]
                            )
                        ),
                        genai.protos.FunctionDeclaration(
                            name="get_live_shuttle_status",
                            description="Searches for shuttle status and routes",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "shuttle_route": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                            enum=[
                                                "Mercado", 
                                                "Polytechnic-Tempe", 
                                                "Tempe-Downtown Phoenix-West", 
                                                "Tempe-West Express", 
                                            ]
                                        ),
                                        description="The Route of Buses"
                                    ),
                                },
                            required= ["shuttle_route"]
                            ),
                        ),
                    ],
                ),
            ],
            tool_config={'function_calling_config': 'ANY'},
        )
        self.firestore = firestore
        self.chat=None
        self.last_request_time = time.time()
        self.request_counter = 0
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        
    async def execute_function(self, function_call):
        """Execute the called function and return its result"""
        func_response = super().execute_function(function_call)
    
        # response = await self.chat.send_message_async(f"{function_name} response : {func_response}")
        if func_response:
            # self._save_message(user_id, "model", f"""(Only Visible to You) System Tools - Discord Agent Response: {func_response}""")
            return func_response
        else:
            self.logger.error(f"Error extracting text from response: {e}")
            return "Error processing response"
        
    def _initialize_model(self):
        if not self.model:
            return self.logger.error("Model not initialized at ActionFunction")
            
        # Rate limiting check
        current_time = time.time()
        if current_time - self.last_request_time < 1.0: 
            raise Exception("Rate limit exceeded")
            
        self.last_request_time = current_time
        self.request_counter += 1
        user_id = discord_state.get("user_id")
        self.chat = self.model.start_chat(history=[],enable_automatic_function_calling=True)


        
    async def determine_action(self, query: str,special_instructions:str) -> str:
        """Determines and executes the appropriate action based on the user query"""
        try:
            self._initialize_model()
            user_id = discord_state.get("user_id")
            final_response = ""
            
            global action_command
            action_command = query

            prompt = f"""
                ### Context:
                - Current Date and Time: {datetime.now().strftime('%H:%M %d') + ('th' if 11<=int(datetime.now().strftime('%d'))<=13 else {1:'st',2:'nd',3:'rd'}.get(int(datetime.now().strftime('%d'))%10,'th')) + datetime.now().strftime(' %B, %Y') }
                - Superior Agent Instruction: {action_command}
                - Superior Agent Remarks: {special_instructions}

                {self.app_config.get_live_status_agent_prompt()}
                
                """

            response = await self.chat.send_message_async(prompt)
            self.logger.info(self._get_chat_history)
            self.logger.info(f"Internal response @ Live Status Model : {response}")
            
            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    
                    final_response = await self.execute_function(part.function_call)
                    self.firestore.update_message("live_status_agent_message", f"Function called {part.function_call}\n Function Response {final_response} ")
                elif hasattr(part, 'text') and part.text.strip():
                    text = part.text.strip()
                    self.firestore.update_message("live_status_agent_message", f"Text Response : {text}")
                    if not text.startswith("This query") and not "can be answered directly" in text:
                        final_response = text.strip()
                        self.logger.info(f"text response : {final_response}")
        
        # Return only the final message
            return final_response if final_response else "Live Status agent fell off! Error 404"
            
        except Exception as e:
            self.logger.error(f"Internal Error @ Live Status Model : {str(e)}")
            return "I apologize, but I couldn't generate a response at this time. Please try again."
        