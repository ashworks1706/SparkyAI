from utils.common_imports import *
class ScholarshipModel:
    
    def __init__(self,firestore,genai,app_config,logger,agent_tools,discord_state):
        self.logger = logger
        self.agent_tools= agent_tools
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
            {self.app_config.get_scholarship_agent_instruction()}
            """,
            tools=[
                genai.protos.Tool(
                    function_declarations=[

                        genai.protos.FunctionDeclaration(
                              name="get_latest_scholarships",
                              description="Fetches comprehensive scholarship information for Arizona State University across various programs.",
                              parameters=content.Schema(
                                  type=content.Type.OBJECT,
                                  properties={
                                      "search_bar_query": content.Schema(
                                          type=content.Type.STRING,
                                          description="General search terms for scholarships"
                                      ),
                                      "academic_level": content.Schema(
                                          type=content.Type.STRING,
                                          description="Academic level of the student",
                                          enum=[
                                              "Graduate",
                                              "Undergraduate"
                                          ]
                                      ),
                                      "citizenship_status": content.Schema(
                                          type=content.Type.STRING,
                                          description="Citizenship status of the applicant",
                                          enum=[
                                              "US Citizen",
                                              "US Permanent Resident", 
                                              "DACA/Dreamer",
                                              "International Student (non-US citizen)"
                                          ]
                                      ),
                                      "gpa": content.Schema(
                                          type=content.Type.STRING,
                                          description="Student's GPA range",
                                          enum=[
                                              "2.0 – 2.24",
                                              "2.25 – 2.49",
                                              "2.50 – 2.74", 
                                              "2.75 - 2.99",
                                              "3.00 - 3.24",
                                              "3.25 - 3.49", 
                                              "3.50 - 3.74",
                                              "3.75 - 3.99",
                                              "4.00"
                                          ]
                                      ),
                                  
                                      "eligible_applicants": content.Schema(
                                          type=content.Type.STRING,
                                          description="Student academic standing",
                                          enum=[
                                              "First-year Undergrads",
                                              "Second-year Undergrads", 
                                              "Third-year Undergrads",
                                              "Fourth-year+ Undergrads",
                                              "Graduate Students",
                                              "Undergraduate Alumni",
                                              "Graduate Alumni"
                                          ]
                                      ),
                                      "focus": content.Schema(
                                          type=content.Type.STRING,
                                          description="Scholarship focus area",
                                          enum=[
                                              "STEM",
                                              "Business and Entrepreneurship",
                                              "Creative and Performing Arts",
                                              "Environment and Sustainability",
                                              "Health and Medicine",
                                              "Social Science",
                                              "International Affairs",
                                              "Public Policy",
                                              "Social Justice",
                                              "Journalism and Media",
                                              "Humanities"
                                          ]
                                      ),
                                      # "college": content.Schema(
                                  #         type=content.Type.STRING,
                                  #         description="ASU College or School",
                                  #         enum=[
                                  #             "Applied Arts and Sciences, School of",
                                  #             "Business, W. P. Carey School of",
                                  #             "Design & the Arts, Herberger Institute for",
                                  #             "Education, Mary Lou Fulton Institute and Graduate School of",
                                  #             "Engineering, Ira A. Fulton Schools of",
                                  #             "Future of Innovation in Society, School for the",
                                  #             "Global Management, Thunderbird School of",
                                  #             "Graduate College",
                                  #             "Health Solutions, College of",
                                  #             "Human Services, College of",
                                  #             "Integrative Sciences and Arts, College of",
                                  #             "Interdisciplinary Arts & Sciences, New College of",
                                  #             "Journalism & Mass Communication, Walter Cronkite School of",
                                  #             "Law, Sandra Day O'Connor College of",
                                  #             "Liberal Arts and Sciences, The College of",
                                  #             "Nursing and Health Innovation, Edson College of",
                                  #             "Public Service and Community Solutions, Watts College of",
                                  #             "Sustainability, School of",
                                  #             "Teachers College, Mary Lou Fulton",
                                  #             "University College"
                                  #         ]
                                  #     ),
                                  }, 
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
            
            'get_latest_scholarships': self.agent_tools.get_latest_scholarships,
        }
        
            
        if function_name in function_mapping:
            function_to_call = function_mapping[function_name]
            func_response = await function_to_call(**function_args)
            # response = await self.chat.send_message_async(f"{function_name} response : {func_response}")
            self.logger.error(f"@scholarship_agent.py Social Media : Function loop response : {func_response}")
            
            if func_response:
                return func_response
            else:
                self.logger.error(f"@scholarship_agent.py Error extracting text from response: {e}")
                return "Error processing response"
            
            
        else:
            raise ValueError(f"Unknown function: {function_name}")
   
        
    def _initialize_model(self):
        if not self.model:
            return self.logger.error(f"@scholarship_agent.py @scholarship_agent.py Model not initialized at ActionFunction")
            
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

                {self.app_config.get_scholarship_agent_prompt()}
                
                """

            response = await self.chat.send_message_async(prompt)
            self.logger.error(f"@scholarship_agent.py @scholarship_agent.py Internal response @ Social Media Model : {response}")
            
            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    
                    final_response = await self.execute_function(part.function_call)
                    self.firestore.update_message("scholarship_agent_message", f"Function called {part.function_call}  Function Response {final_response} ")
                elif hasattr(part, 'text') and part.text.strip():
                    text = part.text.strip()
                    self.firestore.update_message("scholarship_agent_message", f"Text Response : {text}")
                    if not text.startswith("This query") and not "can be answered directly" in text:
                        final_response = text.strip()
                        self.logger.error(f"@scholarship_agent.py @scholarship_agent.py text response : {final_response}")
        
        # Return only the final message
            return final_response if final_response else "Social Media agent fell off! Error 404"
            
        except Exception as e:
            self.logger.error(f"@scholarship_agent.py Internal Error @ Social Media Model : {str(e)}")
            return "I apologize, but I couldn't generate a response at this time. Please try again."
