from utils.common_imports import *
class DataModel:
    def __init__(self, app_config, genai=None, logger=None):
        self.logger = logger
        self.app_config = app_config
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
            {self.app_config.get_data_agent_instruction()}
            """,
            tools=[
            ],
            tool_config={'function_calling_config': 'ANY'},
        )
        self.chat = self.model.start_chat(enable_automatic_function_calling=True)
    async def refine(self, search_context: str, text: str) -> tuple[str, str]:
        prompt = f"""{self.app_config.get_data_agent_prompt()}

        Search Context: {search_context}
        Input Text: {text}

        """

        try:
            self.logger.info(f"Data Model: Refining Data with context : {search_context} \n and data : {text}")
            response = await self.chat.send_message_async(prompt)
            if response and hasattr(response, 'text'):
                parsed = self.parse_json_response(response.text)
                return (
                    parsed.get('document_content', ''),
                    parsed.get('document_title', ''),
                )
            return None, None
        except Exception as e:
            self.logger.error(f"Gemini refinement error: {str(e)}")
            return None, None

    def parse_json_response(self, response_text: str) -> dict:
        """Parse the JSON response into components."""
        try:
            # Remove any potential markdown code block indicators
            cleaned_response = response_text.replace('```json', '').replace('```', '').strip()

            # Parse the JSON string into a dictionary
            parsed_data = json.loads(cleaned_response)

            # Validate required fields
            required_fields = {'document_content', 'document_title'}
            if not all(field in parsed_data for field in required_fields):
                self.logger.error("Missing required fields in JSON response")
                return {'document_content': '', 'document_title': ''}

            return parsed_data

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}")
            return {'document_content': '', 'document_title': ''}
        except Exception as e:
            self.logger.error(f"Unexpected error parsing response: {str(e)}")
            return {'document_content': '', 'document_title': ''}