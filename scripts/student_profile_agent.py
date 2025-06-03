from utils.common_imports import *
class StudentProfileAgent:
    
    def __init__(self,firestore,genai,app_config,logger,student_profile_agent_tools):
        self.logger = logger
        self.agent_tools= student_profile_agent_tools
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
                            name="get_taken_classes",
                            description="Looks at all the classes taken by the student",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "specified_term": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                            enum=[
                                            "Fall 2025",
                                            "Spring 2025",
                                            "Fall 2024",
                                            "Spring 2024",
                                            "Fall 2023",
                                            "Spring 2023",
                                            ]
                                        ),
                                        description=("returns an array of the taken classes by the students with "
                                        "each value of the array as a string formatted (course, grade, term). "
                                        "the specified term parameter can be left empty to list all classes or it can be "
                                        "a specific term to get taken classes on a specific term")
                                    ),
                                },
                                required=["specified_term"]    
                            )
                        ),
                        genai.protos.FunctionDeclaration(
                            name="get_schedule",
                            description="Looks at the current schedule of a student",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "specified_term": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                            enum=[
                                            "Fall 2025",
                                            "Spring 2025",
                                            "Fall 2024",
                                            "Spring 2024",
                                            "Fall 2023",
                                            "Spring 2023",
                                            ]
                                        ),
                                        description=("returns an arraay of all the classes that the student is currently" 
                                        "taking with each value of the array having the format of "
                                        "(course_num, course_title, units, instructor, course_days, course_times, course_dates, course_location),"
                                        "a specific term can be enter in ordered to find a schedule of a specific term or it can"
                                        "be left as an empty string to get the current schedule the student is currently taking")
                                    ),
                                },
                                required=["specified_term"]    
                            )
                        ),
                        genai.protos.FunctionDeclaration(
                            name="get_scholarships",
                            description=("returns an array of all the scholarships the student has recieved where each"
                            "as each value in the array is formatted (award_title, award_amount, term)")
                        ),
                        genai.protos.FunctionDeclaration(
                            name="get_current_charges",
                            description=("returns an array of all the charges applied on the students account"
                            "as each value in the array is formatted (term, due_date, description, amount)")
                        ),
                        genai.protos.FunctionDeclaration(
                            name="get_advisor_info",
                            description=("returns an array of the info of the advisor that is currently assigned to the student"
                            "as each value in the array is formatted (term, due_date, description, amount)")
                        ),
                        genai.protos.FunctionDeclaration(
                            name="quit",
                            description=("quits the webscraping driver that is being used to find all the info about the student")
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
            
            'get_taken_classes': self.agent_tools.get_taken_classes,
            'get_schedule': self.agent_tools.get_schedule,
            'get_scholarships': self.agent_tools.get_scholarships,
            'get_current_charges': self.agent_tools.get_current_charges,
            'get_advisor_info': self.agent_tools.get_advisor_info,
            'quit': self.agent_tools.quit,
        }
        
            
        if function_name in function_mapping:
            function_to_call = function_mapping[function_name]
            func_response = await function_to_call(**function_args)
            # response = await self.chat.send_message_async(f"{function_name} response : {func_response}")
            self.logger.info(f"Student Profile : Function loop response : {func_response}")
            
            if func_response:
                return func_response
            else:
                self.logger.error(f"Error extracting text from response: {e}")
                return "Error processing response"
            
            
        else:
            raise ValueError(f"Unknown function: {function_name}")
   
        
    def _initialize_model(self):
        if not self.model:
            return self.logger.error("Model not initialized at ActionFunction")
            
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

                {self.app_config.get_student_profile_agent_prompt()}
                
                """

            response = await self.chat.send_message_async(prompt)
            self.logger.info(f"Internal response @ Courses Model : {response}")
            
            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    
                    final_response = await self.execute_function(part.function_call)
                    self.firestore.update_message("student_profile_agent_message", f"Function called {part.function_call}\n Function Response {final_response} ")
                elif hasattr(part, 'text') and part.text.strip():
                    text = part.text.strip()
                    self.firestore.update_message("student_profile_agent_message", f"Text Response : {text}")
                    if not text.startswith("This query") and not "can be answered directly" in text:
                        final_response = text.strip()
                        self.logger.info(f"text response : {final_response}")
        
        # Return only the final message
            return final_response if final_response else "Student Profile agent fell off! Error 404"
            
        except Exception as e:
            self.logger.error(f"Internal Error @ Courses Model : {str(e)}")
            return "I apologize, but I couldn't generate a response at this time. Please try again."