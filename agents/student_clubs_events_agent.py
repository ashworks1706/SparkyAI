# student_clubs_events_agent.py

import time
from typing import List, Dict, Any
from datetime import datetime
from urllib.parse import quote_plus

from utils.common_imports import *   # brings in discord, content, HarmCategory, HarmBlockThreshold, etc.

class StudentClubsEventsModel:
    """
    Agent for scraping ASU clubs & events via SunDevilCentral.
    Relies on an injected `genai` client, and two helper tools:
      - get_latest_club_information
      - get_latest_event_updates
    """

    def __init__(
        self,
        middleware,
        genai,                  # <-- the injected GenAI client, *not* a top‐level import
        app_config,
        logger,
        student_clubs_events_agent_tools,
    ):
        self.middleware = middleware
        self.genai = genai
        self.app_config = app_config
        self.logger = logger
        self.agent_tools = student_clubs_events_agent_tools

        # build the Gemini chat with our two functions declared
        self.model = self.genai.GenerativeModel(
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
            system_instruction=self.app_config.get_student_clubs_events_agent_instruction(),
            tools=[
                self.genai.protos.Tool(
                    function_declarations=[
                        self.genai.protos.FunctionDeclaration(
                            name="get_latest_club_information",
                            description="Scrape the ASU student organizations via SunDevilCentral",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "search_bar_query": content.Schema(
                                        type=content.Type.STRING,
                                        description="Optional keyword filter",
                                    ),
                                    "organization_campus": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(type=content.Type.STRING),
                                        description="Optional campus filter",
                                    ),
                                },
                            ),
                        ),
                        self.genai.protos.FunctionDeclaration(
                            name="get_latest_event_updates",
                            description="Scrape the ASU events via SunDevilCentral",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "search_bar_query": content.Schema(
                                        type=content.Type.STRING,
                                        description="Optional keyword filter",
                                    ),
                                    "event_campus": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(type=content.Type.STRING),
                                        description="Optional campus filter",
                                    ),
                                },
                            ),
                        ),
                    ]
                )
            ],
            tool_config={"function_calling_config": "ANY"},
        )

        self.chat = None
        self.last_request_time = time.time()

    def _initialize_model(self):
        """Simple rate‐limit and chat init."""
        now = time.time()
        if now - self.last_request_time < 1.0:
            raise Exception("Rate limit exceeded (1 req/sec)")
        self.last_request_time = now
        self.chat = self.model.start_chat(
            history=[], enable_automatic_function_calling=True
        )

    async def execute_function(self, function_call):
        name = function_call.name
        args = function_call.args

        dispatch = {
            "get_latest_club_information": self.agent_tools.get_latest_club_information,
            "get_latest_event_updates": self.agent_tools.get_latest_event_updates,
        }
        if name not in dispatch:
            raise ValueError(f"Unknown function: {name}")

        result = await dispatch[name](**args)
        return result or f"No results found for `{name}`."

    async def determine_action(self, instruction: str, special_instructions: str) -> str:
        """
        1) Init the chat
        2) Ask Gemini which function to call (if any)
        3) Call it and return its result, or any plain-text answer
        """
        try:
            self._initialize_model()

            prompt = f"""
### Context
- Date & Time: {datetime.now().strftime("%Y-%m-%d %H:%M")}
- Instruction: {instruction}
- Remarks: {special_instructions}

{self.app_config.get_student_clubs_events_agent_prompt()}
"""
            response = await self.chat.send_message_async(prompt)
            final_text = ""

            for part in response.parts:
                if getattr(part, "function_call", None):
                    # Gemini wants us to run one of our tools
                    final_text = await self.execute_function(part.function_call)
                elif getattr(part, "text", "").strip():
                    # A plain AI response
                    final_text = part.text.strip()

            return final_text or "Sorry, I couldn’t find anything."
        except Exception as e:
            self.logger.error(f"StudentClubsEventsModel error: {e}", exc_info=True)
            return "I’m sorry — something went wrong. Please try again later."
