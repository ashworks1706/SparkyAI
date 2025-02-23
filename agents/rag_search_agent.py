from utils.common_imports import *

class RagSearchModel:
    
    def __init__(self, firestore,genai,app_config,logger,rag_search_agent_tools,discord_state,
                 rate_limit_window: float = 1.0, 
                 max_requests: int = 100,
                 retry_attempts: int = 3):
        """
        Initialize RagSearchModel with advanced configuration options.
        
        Args:
            rate_limit_window (float): Time window for rate limiting
            max_requests (int): Maximum number of requests allowed in the window
            retry_attempts (int): Number of retry attempts for function calls
        """
        self.logger=logger
        self.firestore = firestore
        self.app_config = app_config
        self.discord_state= discord_state
        self.agent_tools= rag_search_agent_tools
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
            {self.app_config.get_rag_search_agent_instruction()}
            """,

            tools=[
                genai.protos.Tool(
                    function_declarations=[
                    
                        genai.protos.FunctionDeclaration(
                            name="get_latest_club_information",
                            description="Searches for clubs or organizations information with Sun Devil Search Engine",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "search_bar_query": content.Schema(
                                        type=content.Type.STRING,
                                        description="Search Query",
                                    ),
                                    "organization_campus": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                        ),
                                        description="Club/Organization campus pick from [ASU Downtown, ASU Online, ASU Polytechnic, ASU Tempe, ASU West Valley, Fraternity & Sorority Life, Housing & Residential Life]"
                                    ),
                                    "organization_category": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                        ),
                                        description="Club/Organization Category, pick from [Academic, Barrett, Creative/Performing Arts, Cultural/Ethnic, Distinguished Student Organization, Fulton Organizations, Graduate, Health/Wellness, International, LGBTQIA+, Political, Professional, Religious/Faith/Spiritual, Service, Social Awareness, Special Interest, Sports/Recreation, Sustainability, Technology, Veteran Groups, W.P. Carey Organizations, Women]"
                                    ),
                                },
                                    required=["search_bar_query", "organization_category"]
                            ),
                        ),
                        genai.protos.FunctionDeclaration(
                            name="get_latest_event_updates",
                            description="Searches for events information with Sun Devil Search Engine",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "search_bar_query": content.Schema(
                                        type=content.Type.STRING,
                                        description="Search Query"
                                    ),
                                    "event_campus": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                        ),
                                        description="Event campus pick from [ASU Downtown, ASU Online, ASU Polytechnic, ASU Tempe, ASU West Valley, Fraternity & Sorority Life, Housing & Residential Life]"
                                    ),
                                    "event_category": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                        ),
                                        description="Event Category, pick from [ASU New Student Experience, ASU Sync, ASU Welcome Event, Barrett Student Organization, Career and Professional Development, Club Meetings, Community Service, Cultural, DeStress Fest, Entrepreneurship & Innovation, Graduate, International, Social, Sports/Recreation, Sustainability]"
                                    ),
                                    "event_theme": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                        ),
                                        description="Event Theme, pick from [Arts, Athletics, Community Service, Cultural, Fundraising, GroupBusiness, Social, Spirituality, ThoughtfulLearning]"
                                    ),
                                    "event_perk": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                        ),
                                        description="Event Perk, pick from [Credit, Free Food, Free Stuff]"
                                    ),
                                    "shortcut_date": content.Schema(
                                        type=content.Type.STRING,
                                        description="Event Shortcut date, pick from [tomorrow, this_weekend]"
                                    ),
                                },
                                required=["search_bar_query", "event_category"]
                            ),
                        ),
                        genai.protos.FunctionDeclaration(
                            name="get_latest_news_updates",
                            description="Searches for news information with Sun Devil Search Engine",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "search_bar_query": content.Schema(
                                        type=content.Type.STRING,
                                        description="Search query"
                                    ),
                                    "news_campus": content.Schema(
                                        type=content.Type.STRING,
                                        description="News Campus"
                                    ),
                                },
                                required=["search_bar_query"]
                            ),
                        ),
                        genai.protos.FunctionDeclaration(
                            name="get_latest_sport_updates",
                            description="Fetches comprehensive sports information for Arizona State University across various sports and leagues.",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "search_bar_query": content.Schema(
                                        type=content.Type.STRING,
                                        description="search query to filter sports information"
                                    ),
                                    "sport": content.Schema(
                                        type=content.Type.STRING,
                                        description="Specific sport to search (e.g., 'football', 'basketball', 'baseball', 'soccer')",
                                        enum=[
                                            "football", "basketball", "baseball", 
                                            "soccer", "volleyball", "softball", 
                                            "hockey", "tennis", "track and field"
                                        ]
                                    ),
                                    "league": content.Schema(
                                        type=content.Type.STRING,
                                        description="League for the sport (NCAA, Pac-12, etc.)",
                                        enum=["NCAA", "Pac-12", "Big 12", "Mountain West"]
                                    ),
                                    "match_date": content.Schema(
                                        type=content.Type.STRING,
                                        description="Specific match date in YYYY-MM-DD format"
                                    ),
                                },
                                required=["search_bar_query","sport"]
                            )
                        ),
                        genai.protos.FunctionDeclaration(
                            name="get_latest_social_media_updates",
                            description="Searches for ASU social media posts from specified accounts",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "search_bar_query": content.Schema(
                                        type=content.Type.STRING,
                                        description="Optional search query to filter social media posts"
                                    ),
                                    "account_name": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                            enum=[
                                                "@ArizonaState", 
                                                "@SunDevilAthletics", 
                                                "@SparkySunDevil", 
                                                "@SunDevilFootball", 
                                                "@ASUFootball", 
                                                "@SunDevilFB"
                                            ]
                                        ),
                                        description="Pick from the List of ASU social media account names to search"
                                    )
                                },
                                required=["account_name"]
                            )
                        ),
                        
                        genai.protos.FunctionDeclaration(
                            name="get_library_resources",
                            description="Searches for Books, Articles, Journals, Etc Within ASU Library",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "search_bar_query": content.Schema(
                                        type=content.Type.STRING,
                                        description="search query to filter resources"
                                    ),
                                
                                    "resource_type": content.Schema(
                                        type=content.Type.STRING,
                                        description="Pick Resource Type from the List",
                                        enum=[
                                            "All Items",
                                            "Books",
                                            "Articles",
                                            "Journals",
                                            "Images",
                                            "Scores",
                                            "Maps",
                                            "Sound recordings",
                                            "Video/Film",
                                        ]
                                    ),
                                
                                },
                                required = ["search_bar_query", "resource_type"],
                            )
                        ),
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
                        genai.protos.FunctionDeclaration(
                            name="get_latest_job_updates",
                            description="Searches for jobs from ASU Handshake",
                            parameters=content.Schema(
                                type=content.Type.OBJECT,
                                properties={
                                    "search_bar_query": content.Schema(
                                        type=content.Type.STRING,
                                        description="Optional search query to filter jobs"
                                    ),
                                    "job_type": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                            enum=[
                                                "Full-Time", 
                                                "Part-Time", 
                                                "Internship", 
                                                "On-Campus"
                                            ]
                                        ),
                                        description="Pick from the List of Job Types to search"
                                    ),
                                    "job_location": content.Schema(
                                        type=content.Type.ARRAY,
                                        items=content.Schema(
                                            type=content.Type.STRING,
                                            enum=[
                                                "Tempe, Arizona, United States",

                                                "Mesa, Arizona, United States",

                                                "Phoenix, Arizona, United States",
                                            ]
                                        ),
                                        description="Pick from the List of ASU Locations to search"
                                    ),
                                },
                                
                            )
                        ),
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
                                    "subject_name": content.Schema(
                                        type=content.Type.STRING,
                                        description="""Class/Course Name """,
                                        
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
                                    
                            )
                        ),
                    ],
                ),
            ],
            tool_config={'function_calling_config': 'AUTO'},
        )
        self.chat = None
        self.last_request_time = time.time()
        self.request_counter = 0
        self.rate_limit_window = rate_limit_window
        self.max_requests = max_requests
        self.retry_attempts = retry_attempts
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        self.logger.info(f"RagSearchModel initialized with rate limit: {rate_limit_window}s, max requests: {max_requests}")

    async def execute_function(self, function_call):
        """
        Execute the called function with comprehensive error handling and retry mechanism.
        
        Args:
            function_call: Function call OBJECT to execute
        
        Returns:
            str: Processed function response
        """
        function_name = function_call.name
        function_args = function_call.args
        
        function_mapping = {
            'get_latest_club_information': self.agent_tools.get_latest_club_information,
            'get_latest_event_updates': self.agent_tools.get_latest_event_updates,
            'get_latest_news_updates': self.agent_tools.get_latest_news_updates,
            'get_latest_social_media_updates': self.agent_tools.get_latest_social_media_updates,
            'get_latest_sport_updates': self.agent_tools.get_latest_sport_updates,
           'get_library_resources': self.agent_tools.get_library_resources,
              'get_latest_scholarships': self.agent_tools.get_latest_scholarships,
            'get_latest_job_updates': self.agent_tools.get_latest_job_updates,
            'get_latest_class_information': self.agent_tools.get_latest_class_information
        }
        
        if function_name not in function_mapping:
            self.logger.error(f"Unknown function: {function_name}")
            raise ValueError(f"Unknown function: {function_name}")
        
        function_to_call = function_mapping[function_name]
        
        for attempt in range(self.retry_attempts):
            try:
                func_response = await function_to_call(**function_args)
                
                self.logger.info(f"Function '{function_name}' response (Attempt {attempt + 1}): {func_response}")
                
                if func_response:
                    return func_response
                
                self.logger.warning(f"Empty response from {function_name}")
                
            except Exception as e:
                self.logger.error(f"Function call error (Attempt {attempt + 1}): {str(e)}")
                
                if attempt == self.retry_attempts - 1:
                    self.logger.critical(f"All retry attempts failed for {function_name}")
                    return f"Error processing {function_name}"
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        return "No valid response from function"
        
    def _initialize_model(self):
        """
        Initialize the search model with advanced rate limiting and error checking.
        """
        if not self.model:
            self.logger.critical("Model not initialized")
            raise RuntimeError("Search model is not configured")
        
        current_time = time.time()
        
        if current_time - self.last_request_time < self.rate_limit_window:
            self.request_counter += 1
            if self.request_counter > self.max_requests:
                wait_time = self.rate_limit_window - (current_time - self.last_request_time)
                self.logger.warning(f"Rate limit exceeded. Waiting {wait_time:.2f} seconds")
                raise Exception(f"Rate limit exceeded. Please wait {wait_time:.2f} seconds")
        else:
            # Reset counter if outside the rate limit window
            self.request_counter = 1
            self.last_request_time = current_time
        
        try:
            user_id = self.discord_state.get('user_id')
            self.chat = self.model.start_chat(history=[],enable_automatic_function_calling=True)
            self.logger.info("\nSearch model chat session initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize chat session: {str(e)}")
            raise RuntimeError("Could not start chat session")
        

        
    async def determine_action(self,instruction_to_agent:str,special_instructions:str) -> str:
        """
        Advanced query processing with comprehensive error handling and logging.
        
        Args:
            query (str): User query to process
        
        Returns:
            str: Processed query response
        """
        try:
            user_id = self.discord_state.get("user_id")
            self._initialize_model()
            final_response = ""
            
            prompt = f"""
             ### Context:
                - Current Date and Time: {datetime.now().strftime('%H:%M %d') + ('th' if 11<=int(datetime.now().strftime('%d'))<=13 else {1:'st',2:'nd',3:'rd'}.get(int(datetime.now().strftime('%d'))%10,'th')) + datetime.now().strftime(' %B, %Y') }
                - Superior Agent Instruction: {instruction_to_agent}
                - Superior Agent Remarks: {special_instructions}
                {self.app_config.get_rag_search_agent_prompt()}
                
                """
                
            self.logger.debug(f"Generated prompt: {prompt}")
            
            try:
                response = await self.chat.send_message_async(prompt)
                for part in response.parts:
                    if hasattr(part, 'function_call') and part.function_call: 
                        final_response = await self.execute_function(part.function_call)
                        self.firestore.update_message("rag_search_agent_message", f"Function called {part.function_call}\n Function Response {final_response} ")
                    elif hasattr(part, 'text') and part.text.strip():
                        text = part.text.strip()
                        self.firestore.update_message("rag_search_agent_message", f"Text Response : {text} ")
                        if not text.startswith("This query") and "can be answered directly" not in text:
                            final_response = text.strip()
            
            except Exception as response_error:
                self.logger.error(f"Response generation error: {str(response_error)}")
                final_response = "Unable to generate a complete response"
            
            return final_response or "Search agent encountered an unexpected issue"
        
        except Exception as critical_error:
            self.logger.critical(f"Critical error in determine_action: {str(critical_error)}")
            return "I'm experiencing technical difficulties. Please try again later."
        