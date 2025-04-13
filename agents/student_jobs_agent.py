from utils.common_imports import *
from typing import Dict, List

class StudentJobsModel:
    
    def __init__(self, firestore, genai, app_config, logger, jobs_agent_tools, discord_state):
        self.logger = logger
        self.agent_tools = jobs_agent_tools
        self.discord_state = discord_state
        self.app_config = app_config

        # Create the Gemini model with a new function declaration "get_workday_student_jobs"
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
            system_instruction=f"""
            {self.app_config.get_student_jobs_agent_instruction()}
            """,
            tools=[
                genai.protos.Tool(
                    function_declarations=[
                        genai.protos.FunctionDeclaration(
                            name="get_workday_student_jobs",
                            description="Scrapes top jobs on ASU Workday for a given keyword",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "keyword": content.Schema(
                                        type=content.Type.STRING,
                                        description="Keyword to search on Workday (e.g. 'marketing')"
                                    ),
                                    "max_results": content.Schema(
                                        type=content.Type.INTEGER,
                                        description="Max number of jobs to return"
                                    ),
                                },
                                required=["keyword"]
                            )
                        ),
                    ],
                ),
            ],
            tool_config={'function_calling_config': 'ANY'},
        )

        self.firestore = firestore
        self.chat = None
        self.last_request_time = time.time()
        self.request_counter = 0
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        
    async def execute_function(self, function_call):
        """Execute whichever function the agent has decided to call."""
        function_name = function_call.name
        function_args = function_call.args
        
        function_mapping = {
            'get_workday_student_jobs': self.agent_tools.get_workday_student_jobs,
        }
        
        if function_name in function_mapping:
            function_to_call = function_mapping[function_name]
            func_response = await function_to_call(**function_args)
            self.logger.info(f"@student_jobs_agent.py: Function '{function_name}' response: {func_response}")
            
            if func_response:
                return func_response
            else:
                self.logger.error("@student_jobs_agent.py Error: function returned no data.")
                return "Error processing response"
        else:
            raise ValueError(f"Unknown function: {function_name}")
   
    def _initialize_model(self):
        """Setup or check the model, plus basic rate-limiting logic."""
        if not self.model:
            self.logger.error("@student_jobs_agent.py Model not initialized.")
            return

        # Rate limiting check
        current_time = time.time()
        if current_time - self.last_request_time < 1.0: 
            raise Exception("Rate limit exceeded")
            
        self.last_request_time = current_time
        self.request_counter += 1
        self.chat = self.model.start_chat(history=[], enable_automatic_function_calling=True)

    async def determine_action(self, instruction_to_agent: str, special_instructions: str) -> str:
        """
        Invoked with the user's query. The agent uses its context and the prompt 
        to determine if it should call get_workday_student_jobs, then returns the result.
        """
        try:
            self._initialize_model()
            final_response = ""

            prompt = f"""
                ### Context:
                - Current Date and Time: {datetime.now().strftime('%H:%M %d') + (
                    'th' if 11 <= int(datetime.now().strftime('%d')) <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(int(datetime.now().strftime('%d')) % 10, 'th')
                ) + datetime.now().strftime(' %B, %Y')}
                - Superior Agent Instruction: {instruction_to_agent}
                - Superior Agent Remarks: {special_instructions}

                {self.app_config.get_student_jobs_agent_prompt()}
            """

            response = await self.chat.send_message_async(prompt)
            self.logger.info(f"@student_jobs_agent.py: Raw agent response: {response}")
            
            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    final_response = await self.execute_function(part.function_call)
                    self.firestore.update_message(
                        "jobs_agent_message", 
                        f"Function called {part.function_call}  Function Response {final_response}"
                    )
                elif hasattr(part, 'text') and part.text.strip():
                    text = part.text.strip()
                    self.firestore.update_message("jobs_agent_message", f"Text Response: {text}")
                    if not text.startswith("This query") and "can be answered directly" not in text:
                        final_response = text
                        self.logger.info(f"@student_jobs_agent.py: Plain text response: {final_response}")

            return final_response if final_response else "Jobs agent fell off! Error 404"
            
        except Exception as e:
            self.logger.error(f"@student_jobs_agent.py: Internal Error: {str(e)}")
            return "I apologize, but I couldn't generate a response at this time. Please try again."
