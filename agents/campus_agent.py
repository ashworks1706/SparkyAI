from utils.common_imports import *
import time

class CampusAgentModel:
    def __init__(self, middleware, genai, app_config, logger, campus_agent_tools):
        self.middleware        = middleware
        self.genai             = genai
        self.app_config        = app_config
        self.logger            = logger
        self.agent_tools       = campus_agent_tools
        self.chat              = None
        self.last_request_time = time.time()

        # Gemini model setup with single get_campus_location tool
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.0,
                "top_p": 0.1,
                "top_k": 40,
                "max_output_tokens": 500,
                "response_mime_type": "text/plain",
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            },
            system_instruction=self.app_config.get_campus_agent_instruction(),
            tools=[
                genai.protos.Tool(
                    function_declarations=[
                        genai.protos.FunctionDeclaration(
                            name="get_campus_location",
                            description="Lookup a building or landmark on ASU's interactive campus map",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "keyword": content.Schema(
                                        type=content.Type.STRING,
                                        description="Building code or landmark (e.g. 'PSH', 'LSE')"
                                    )
                                },
                                required=["keyword"]
                            )
                        )
                    ]
                )
            ],
            tool_config={"function_calling_config": "ANY"},
        )

    def _initialize_model(self):
        """Simple rate‐limit and start a fresh chat."""
        now = time.time()
        if now - self.last_request_time < 1.0:
            raise Exception("Rate limit exceeded for campus agent")
        self.last_request_time = now
        self.chat = self.model.start_chat(history=[], enable_automatic_function_calling=True)

    async def execute_function(self, function_call):
        """Dispatch our single map‐lookup tool."""
        if function_call.name == "get_campus_location":
            return await self.agent_tools.get_campus_location(**function_call.args)
        raise ValueError(f"Unknown function: {function_call.name}")

    async def determine_action(self, instruction_to_agent: str, special_instructions: str) -> str:
        """
        Either returns plain text or invokes get_campus_location() and returns its output.
        """
        try:
            self._initialize_model()

            prompt = f"""
### Context:
- User Request: {instruction_to_agent}
- Notes: {special_instructions}

{self.app_config.get_campus_agent_prompt()}
"""
            response = await self.chat.send_message_async(prompt)
            final_answer = None

            for part in response.parts:
                # tool call
                if hasattr(part, "function_call") and part.function_call:
                    res = await self.execute_function(part.function_call)
                    # log into middleware (ensure your schema allows this key)
                    self.middleware.update_message(
                        "campus_agent_message",
                        f"FunctionCalled get_campus_location → {res}"
                    )
                    final_answer = res

                # or plain text
                elif hasattr(part, "text") and part.text.strip():
                    text = part.text.strip()
                    self.middleware.update_message("campus_agent_message", f"TextResponse: {text}")
                    final_answer = text

            return final_answer or "Campus agent couldn't produce an answer."
        except Exception as e:
            self.logger.error(f"campus_agent.py error: {e}", exc_info=True)
            return "I’m sorry, I couldn’t look up that location right now."
