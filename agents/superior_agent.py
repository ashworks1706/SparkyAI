from utils.common_imports import *
class SuperiorModel:
    
    def __init__(self, firestore,genai,app_config,logger,superior_agent_tools):
        self.logger = logger
        self.firestore = firestore
        self.app_config = app_config
        self.agent_tools=superior_agent_tools
        self.model = genai.GenerativeModel(model_name="gemini-2.0-flash",
        
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
                    name="access_rag_search_agent",
                    description="Performs RAG Search to provide rapid result summary",
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
                
                genai.protos.FunctionDeclaration(
                    name="send_bot_feedback",
                    description="Submits any feedback or errors about sparky to mods",
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
                    name="access_shuttle_status_agent",
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
                    name="access_courses_agent",
                    description="Has ability to search for ASU courses information",
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
                    name="access_student_clubs_events_agent",
                    description="Has ability to search for ASU events information and ASU student clubs information",
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
                    name="access_library_agent",
                    description="Has ability to search for ASU library information",
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
                    name="access_news_media_agent",
                    description="Has ability to search for ASU news and ASU Social Media information",
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
                    name="access_scholarship_agent",
                    description="Has ability to search for ASU scholarship information",
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
                    name="access_sports_agent",
                    description="Has ability to search for ASU sports information",
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
                    name="access_student_jobs_agent",
                    description="Has ability to search for ASU student jobs information",
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

               
                ],
            ),
            ],
            tool_config={'function_calling_config': 'AUTO'},
        )
        self.chat = None
        
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
            'access_rag_search_agent': self.agent_tools.access_rag_search_agent,
            'access_discord_agent': self.agent_tools.access_discord_agent,
            'access_shuttle_status_agent': self.agent_tools.access_shuttle_status_agent,
            'get_user_profile_details': self.agent_tools.get_user_profile_details,
            'send_bot_feedback': self.agent_tools.send_bot_feedback,
            'access_courses_agent': self.agent_tools.access_courses_agent,
            'access_student_clubs_events_agent': self.agent_tools.access_student_clubs_events_agent,
            'access_library_agent': self.agent_tools.access_library_agent,
            'access_news_media_agent': self.agent_tools.access_news_media_agent,
            'access_scholarship_agent': self.agent_tools.access_scholarship_agent,
            'access_student_jobs_agent': self.agent_tools.access_student_jobs_agent,
        }

        function_name = function_call.name
        function_args = function_call.args

        if function_name not in function_mapping:
            raise ValueError(f"Unknown function: {function_name}")
        
        function_to_call = function_mapping[function_name]
        self.logger.info(f"@superior_agent.py Function called : {function_name}")
        return await function_to_call(**function_args)

    async def process_gemini_response(self, response: Any) -> tuple[str, bool, Any]:
        text_response = ""
        has_function_call = False
        function_call = None
        
        for part in response.parts:
            if hasattr(part, 'text') and part.text.strip():
                text_response += f"\n{part.text.strip()}"
                self.logger.info(f"@superior_agent.py text response : {text_response}")
                self.firestore.update_message("superior_agent_message", f"Text Response : {text_response} ")
            if hasattr(part, 'function_call') and part.function_call:
                has_function_call = True
                function_call = part.function_call
                self.logger.info(f"@superior_agent.py function response : {part.function_call}")
                temp_func =  {
                "function_call": {
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args)
                    }
                }
                self.logger.info(f"@superior_agent.py @ Superior Agent formed function call : {temp_func}")
                self.firestore.update_message("superior_agent_message", f"temp_func")
                self.logger.info(f"@superior_agent.py @superior_agent.py Updated Firestore message")
        self.logger.info(f"@superior_agent.py @Superior Agent : text_Response :{text_response}\n has_function_call {has_function_call}\n function_call {function_call} ")
        return text_response, has_function_call, function_call

    async def determine_action(self, query: str) -> List[str]:
        try:
            max_depth=3
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
            max_depth-=1
            
            while True:
                text_response, has_function_call, function_call = await self.process_gemini_response(response)
                responses.append(text_response)
                final_response += text_response
                if not has_function_call and "send_bot_feedback" not in final_response:
                    self.logger.info(f"@superior_agent.py @superior_agent.py @ Superior Agent : not function call requested")
                    break
                if not has_function_call and "send_bot_feedback" in final_response:
                    self.logger.info(f"@superior_agent.py @superior_agent.py @ Superior Agent : not function call requested")
                    break
                self.logger.info(f"@superior_agent.py @superior_agent.py @ Superior Agent : function call requested")
                function_result = await self.execute_function(function_call)
                self.firestore.update_message("superior_agent_message", f"""(User cannot see this response) System Generated - \n{function_call.name}\nResponse: {function_result}\nAnalyze the response and answer the user's question. Feel free to use more functions inorder to answer question.""")
                self.logger.info(f"@superior_agent.py @superior_agent.py \nAction Model @ Function result is: %s", function_result)
                response = await self.chat.send_message_async(f"""(User cannot see this response) System Generated - \n{function_call.name}\nResponse: {function_result}\nAnalyze the response and answer the user's question. Feel free to use more functions inorder to answer question. Remaining function tries : {max_depth}""")
                max_depth-=1
                
            final_response = " ".join(response.strip() for response in responses if response.strip())
            
            return final_response.strip()
        
        except Exception as e:
            self.logger.error(f"@superior_agent.py Error in determine_action: {e}")
            return ["I'm sorry, I couldn't generate a response. Please try again."]