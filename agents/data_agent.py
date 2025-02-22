class DataModel:
    def __init__(self, model=None):
        self.model = model
  
    def refine(self, search_context: str, text: str) -> tuple[str, str, str]:
        prompt = f"""{app_config.get_data_agent_prompt()}        
        Search Context: {search_context}
        Input Text: {text}

        """

        try:
            logger.info(f"Data Model: Refining Data with context : {search_context} \n and data : {text}")
            response = self.model.generate_content(prompt)
            if response and hasattr(response, 'text'):
                parsed = self.parse_json_response(response.text)
                return (
                    # parsed.get('refined_content', ''),
                    parsed.get('title', ''),
                )
            return None, None, None
        except Exception as e:
            logger.error(f"Gemini refinement error: {str(e)}")
            return None, None, None

    def parse_json_response(self, response_text: str) -> dict:
        """Parse the JSON response into components."""
        try:
            # Remove any potential markdown code block indicators
            cleaned_response = response_text.replace('```json', '').replace('```', '').strip()

            # Parse the JSON string into a dictionary
            parsed_data = json.loads(cleaned_response)

            # Validate required fields
            required_fields = { 'title'}
            if not all(field in parsed_data for field in required_fields):
                logger.error("Missing required fields in JSON response")
                return { 'title': ''}

            return parsed_data

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return { 'title': '', 'category': ''}
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {str(e)}")
            return {'title': '', 'category': ''}