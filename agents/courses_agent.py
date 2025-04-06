from utils.common_imports import *
class CoursesModel:
    
    def __init__(self,firestore,genai,app_config,logger,courses_agent_tools,discord_state):
        self.logger = logger
        self.agent_tools= courses_agent_tools
        self.discord_state = discord_state
        self.app_config= app_config
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
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
            {self.app_config.get_courses_agent_instruction()}
            """,
            tools=[
                genai.protos.Tool(
                    function_declarations=[

                        genai.protos.FunctionDeclaration(
                            name="get_latest_class_information",
                            description="Searches for ASU Classes information indepth with ASU Catalog Search",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "search_bar_query": content.Schema(
                                        type=content.Type.STRING,
                                        description=" search query to filter classes"
                                    ),
                                
                                    "class_seat_availability": content.Schema(
                                        type=content.Type.STRING,
                                        description="Pick from the List of classAvailability : Open | All",
                                        enum=[
                                            "Open",
                                            "All"
                                        ]
                                    ),
                                    "class_term": content.Schema(
                                        type=content.Type.STRING,
                                        description="Pick from this list from the list",
                                        enum=[
                                        "Fall 2026",
                                        "Summer 2026",
                                        "Spring 2026",
                                        "Fall 2025",
                                        "Summer 2025",
                                        "Spring 2025",
                                        ]

                                    ),
                              
                                    
                                    "num_of_credit_units": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                            enum=[
                                                "1", 
                                                "2", 
                                                "3", 
                                                "4", 
                                                "5", 
                                                "6",
                                                "7",
                                            ]
                                        ),
                                        description="Pick from the List of from classCredits"
                                    ),
                                    "class_session": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                            enum=[
                                                "A", 
                                                "B", 
                                                "C", 
                                                "Other", 
                                            ]
                                        ),
                                        description="Pick from the List of from classSessions"
                                    ),
                                    "class_days": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                            enum=[
                                                "Monday", 
                                                "Tuesday", 
                                                "Wednesday", 
                                                "Thursday", 
                                                "Friday", 
                                                "Saturday", 
                                                "Sunday", 
                                            ]
                                        ),
                                        description="Pick from the List of from classSessions"
                                    ),
                                    "class_location": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                            enum=[
                                                "TEMPE",
                                                "WEST",
                                                "POLY",
                                                "OFFCAMP",
                                                "PHOENIX",
                                                "LOSANGELES",
                                                "CALHC",
                                                "ASUSYNC",
                                                "ASUONLINE",
                                                "ICOURSE"
                                                ]
                                        ),
                                        description="Pick from the List of from classLocations"
                                    ),
                                
                                },
                                 required=["search_bar_query"]
                                    
                            )
                            
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
        function_name = function_call.name
        function_args = function_call.args
        
        function_mapping = {
            
            'get_latest_class_information': self.agent_tools.get_latest_class_information,
        }
        
            
        if function_name in function_mapping:
            function_to_call = function_mapping[function_name]
            func_response = await function_to_call(**function_args)
            # response = await self.chat.send_message_async(f"{function_name} response : {func_response}")
            self.logger.info("@courses_agent_tools.py Courses : Function loop response : {func_response}")
            
            if func_response:
                return func_response
            else:
                self.logger.error(f"@courses_agent_tools.py Error extracting text from response: {e}")
                return "Error processing response"
            
            
        else:
            raise ValueError(f"Unknown function: {function_name}")
   
        
    def _initialize_model(self):
        if not self.model:
            return self.logger.error("@courses_agent_tools.py Model not initialized at ActionFunction")
            
        # Rate limiting check
        current_time = time.time()
        if current_time - self.last_request_time < 1.0: 
            raise Exception("Rate limit exceeded")
            
        self.last_request_time = current_time
        self.request_counter += 1
        user_id = self.discord_state.get("user_id")
        self.chat = self.model.start_chat(history=[],enable_automatic_function_calling=True)


        
    async def determine_action(self,instruction_to_agent:str,special_instructions:str) -> str:
        """Determines and executes the appropriate action based on the user query"""
        try:
            self._initialize_model()
            user_id = self.discord_state.get("user_id")
            final_response = ""
            

            prompt = f"""
                ### Context:
                - Current Date and Time: {datetime.now().strftime('%H:%M %d') + ('th' if 11<=int(datetime.now().strftime('%d'))<=13 else {1:'st',2:'nd',3:'rd'}.get(int(datetime.now().strftime('%d'))%10,'th')) + datetime.now().strftime(' %B, %Y') }
                - Superior Agent Instruction: {instruction_to_agent}
                - Superior Agent Remarks: {special_instructions}

                {self.app_config.get_courses_agent_prompt()}
                
                """

            response = await self.chat.send_message_async(prompt)
            self.logger.info("@courses_agent_tools.py Internal response @ Courses Model : {response}")
            
            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    
                    final_response = await self.execute_function(part.function_call)
                    self.firestore.update_message("courses_agent_message", f"Function called {part.function_call}\n Function Response {final_response} ")
                elif hasattr(part, 'text') and part.text.strip():
                    text = part.text.strip()
                    self.firestore.update_message("courses_agent_message", f"Text Response : {text}")
                    if not text.startswith("This query") and not "can be answered directly" in text:
                        final_response = text.strip()
                        self.logger.info("@courses_agent_tools.py text response : {final_response}")
        
        # Return only the final message
            return final_response if final_response else "Courses agent fell off! Error 404"
            
        except Exception as e:
            self.logger.error(f"@courses_agent_tools.py Internal Error @ Courses Model : {str(e)}")
            return "I apologize, but I couldn't generate a response at this time. Please try again."
