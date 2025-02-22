class AgentTools:
    def __init__(self,firestore,discord_state,utils,app_config, live_status_agent,rag_rag_search_agent,discord_agent):
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3
        self.discord_state = discord_state
        self.utils = utils
        self.firestore = firestore
        self.discord_client = self.discord_state.get('discord_client')
        logger.info(f"Initialized Discord Client : {self.discord_client}")
        self.guild = self.discord_state.get('target_guild')
        self.user_id=self.discord_state.get('user_id')
        self.user=self.discord_state.get('user')
        logger.info(f"Initialized Discord Guild : {self.guild}")
        self.conversations = {}
        self.app_config= app_config
        self.client = genai_vertex.Client(api_key=self.app_config.get_api_key())
        self.model_id = "gemini-2.0-flash-exp"
        self.google_search_tool = Tool(google_search=GoogleSearch())    
                
    async def get_live_library_status(self, status_type : [] = None, date : str = None, library_names: [] = None):
        """
        Retrieve ASU Library Status using ASU Library Search with robust parameter handling.

        Args:
            status_type ([]): Open or Close & Study Room Availability.
            library_names ([], optional): Name of library.
            date (str): Dec 01 (Month_prefix + Date).

        Returns:
            Union[str, dict]: Search results or error message.
            
        Notes:
        Example URL- 
        https://asu.libcal.com/r/accessible/availability?lid=13858&gid=28619&zone=0&space=0&capacity=2&date=2024-12-04
        """
        
        if not (library_names or status_type or date):
            return "Error: Atleast one parameter required"
        
        search_url=None
        query=None
        result =""
        doc_title = " ".join(library_names)
        if "Availability" in status_type:
            search_url=f"https://lib.asu.edu/hours"
            query = f"library_names={library_names}&date={date}"
            result+=await self.utils.perform_web_search(search_url, query,doc_title=doc_title, doc_category ="libraries_status")
        
        library_map = {
            "Tempe Campus - Hayden Library": "13858",
            "Tempe Campus - Noble Library": "1702",
            "Downtown Phoenix Campus - Fletcher Library": "1703",
            "West Campus - Library": "1707",
            "Polytechnic Campus - Library": "1704"
        }
        
        gid_map={
            "13858": "28619",
             "1702": "2897",
            "1703": "2898",
            "1707": "28611",
            "1704": "2899"
        }
             
        if "StudyRoomsAvailability" in status_type:
            transformed_date = datetime.strptime(date, '%b %d').strftime('2024-%m-%d')
            for library in library_names:
                query= library_map[library]
                search_url = f"https://asu.libcal.com/r/accessible/availability?lid={library_map[library]}&gid={gid_map[library_map[library]]}&zone=0&space=0&capacity=2&date={transformed_date}"
                result+=await self.utils.perform_web_search(search_url, query,doc_title=doc_title, doc_category ="libraries_status")
            
        return result
        
    async def get_live_shuttle_status(self, shuttle_route: [] = None):
        if not shuttle_route:
            return "Error: At least one route is required"
        
        shuttle_route = set(shuttle_route)
        
        doc_title = " ".join(shuttle_route)
        search_url="https://asu-shuttles.rider.peaktransit.com/"

        logger.info(shuttle_route)
        
        if len(shuttle_route) == 1:
            logger.info("\nOnly one route")
            route = next(iter(shuttle_route))
            return await self.utils.perform_web_search(search_url, optional_query=route,doc_title=doc_title, doc_category ="shuttles_status")

        # Multiple routes handling
        result = ""
        try:
            for route in shuttle_route:
                result += await self.utils.perform_web_search(search_url, optional_query=route,doc_title=doc_title, doc_category ="shuttles_status")
            logger.info("\nDone")
            return result
        except Exception as e:
            return f"Error performing shuttle search: {str(e)}"
                
    async def get_latest_club_information(self, search_bar_query: str = None, organization_category: list = None, organization_campus: list = None):
        if not any([search_bar_query, organization_category, organization_campus]):
            return "At least one parameter of this function is required. Neither Search query and organization category and organization campus received. Please provide at least one parameter to perform search."
        
        search_url = "https://asu.campuslabs.com/engage/organizations"
        params = []
        organization_campus_ids = { "ASU Downtown":"257211",
                                   "ASU Online":"257214",
                                   "ASU Polytechnic":"257212",
                                   "ASU Tempe":"254417",
                                   "ASU West Valley":"257213",
                                   "Fraternity & Sorority Life":"257216",
                                   "Housing & Residential Life":"257215"}
        
        organization_category_ids = {"Academic":"13382","Barrett":"14598","Creative/Performing Arts":"13383","Cultural/Ethnic":"13384","Distinguished Student Organization":"14549","Fulton Organizations":"14815","Graduate":"13387","Health/Wellness":"13388","International":"13389","LGBTQIA+":"13391","Political":"13392","Professional":"13393","Religious/Faith/Spiritual":"13395","Service":"13396","Social Awareness":"13398","Special Interest":"13399","Sports/Recreation":"13400","Sustainability":"13402","Technology":"13403", "Veteran Groups":"14569","W.P. Carey Organizations":"14814","Women":"13405"}
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif organization_category:
            doc_title = " ".join(organization_category)
        elif organization_campus:
            doc_title = " ".join(organization_campus)
        else:
            doc_title = None

 
        if organization_campus:
            campus_id_array = [organization_campus_ids[campus] for campus in organization_campus if campus in organization_campus_ids]
            if campus_id_array:
                params.extend([f"branches={campus_id}" for campus_id in campus_id_array])
        
        if organization_category:
            category_id_array = [organization_category_ids[category] for category in organization_category if category in organization_category_ids]
            if category_id_array:
                params.extend([f"categories={category_id}" for category_id in category_id_array])
        
        if search_bar_query:
            params.append(f"query={search_bar_query.lower().replace(' ', '%20')}")
        
        if params:
            search_url += "?" + "&".join(params)
        
        return await self.utils.perform_web_search(search_url, doc_title=doc_title, doc_category ="clubs_info")
        
    async def get_latest_event_updates(self, search_bar_query: str = None, event_category: list = None, 
                               event_theme: list = None, event_campus: list = None, 
                               shortcut_date: str = None, event_perk: list = None):
        
        if not any([search_bar_query, event_category, event_theme, event_campus]):
            return "At least one parameter of this function is required. Neither Search query and organization category and organization campus received. Please provide at least one parameter to perform search."
        
        search_url = "https://asu.campuslabs.com/engage/events"
        params = []
        
        event_campus_ids = {
            "ASU Downtown": "257211",
            "ASU Online": "257214",
            "ASU Polytechnic": "257212",
            "ASU Tempe": "254417",
            "ASU West Valley": "257213",
            "Fraternity & Sorority Life": "257216",
            "Housing & Residential Life": "257215"
        }
        
        event_category_ids = {
            "ASU New Student Experience": "18002",
            "ASU Sync": "15695",
            "ASU Welcome Event": "12897",
            "Barrett Student Organization": "12902",
            "Black History Month": "21730",
            "C3": "19049",
            "Career and Professional Development": "12885",
            "Change The World": "12887",
            "Changemaker Central": "12886",
            "Civic Engagement": "17075",
            "Club Meetings": "12887",
            "Clubs and Organization Information": "12888",
            "Community Service": "12903",
            "Cultural Connections and Multicultural community of Excellence": "21719",
            "Culture @ ASU": "12898",
            "DeStress Fest": "19518",
            "Entrepreneurship & Innovation": "17119",
            "General": "12889",
            "Graduate": "12906",
            "Hispanic Heritage Month": "21723",
            "Homecoming": "20525",
            "In-Person Event": "17447",
            "International": "12899",
            "Memorial Union & Student Pavilion Programs": "12900",
            "Multicultural community of Excellence": "19389",
            "PAB Event": "12890",
            "Salute to Service": "12891",
            "Student Engagement Event": "12892",
            "Student Organization Event": "12893",
            "Sun Devil Athletics": "12894",
            "Sun Devil Civility": "12901",
            "Sun Devil Fitness/Wellness": "12895",
            "Sustainability": "12905",
            "University Signature Event": "12904",
            "W.P. Carey Event": "17553"
        }
        
        event_theme_ids = {
            "Arts": "arts",
            "Athletics": "athletics",
            "Community Service": "community_service",
            "Cultural": "cultural",
            "Fundraising": "fundraising",
            "GroupBusiness": "group_business",
            "Social": "social",
            "Spirituality": "spirituality",
            "ThoughtfulLearning": "thoughtful_learning"
        }
        
        event_perk_ids = {
            "Credit": "Credit",
            "Free Food": "FreeFood",
            "Free Stuff": "FreeStuff"
        }
        
        if event_campus:
            campus_id_array = [event_campus_ids[campus] for campus in event_campus if campus in event_campus_ids]
            if campus_id_array:
                params.extend([f"branches={campus_id}" for campus_id in campus_id_array])
        
        if event_category:
            category_id_array = [event_category_ids[category] for category in event_category if category in event_category_ids]
            if category_id_array:
                params.extend([f"categories={category_id}" for category_id in category_id_array])
        
        if event_theme:
            theme_id_array = [event_theme_ids[theme] for theme in event_theme if theme in event_theme_ids]
            if theme_id_array:
                params.extend([f"themes={theme_id}" for theme_id in theme_id_array])
        
        if event_perk:
            perk_id_array = [event_perk_ids[perk] for perk in event_perk if perk in event_perk_ids]
            if perk_id_array:
                params.extend([f"perks={perk_id}" for perk_id in perk_id_array])
        
        if shortcut_date:
            valid_dates = ["tomorrow", "this_weekend"]
            if shortcut_date.lower() in valid_dates:
                params.append(f"shortcutdate={shortcut_date.lower()}")
        
        if search_bar_query:
            params.append(f"query={search_bar_query.lower().replace(' ', '%20')}")
        
        if params:
            search_url += "?" + "&".join(params)
        
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif event_category:
            doc_title = " ".join(event_category)
        elif event_theme:
            doc_title = " ".join(event_theme)
        elif event_campus:
            doc_title = " ".join(event_campus)
        elif shortcut_date:
            doc_title = shortcut_date
        elif event_perk:
            doc_title = " ".join(event_perk)
        else:
            doc_title = None
        
        return await self.utils.perform_web_search(search_url, doc_title=doc_title, doc_category = "events_info")
        
    async def get_latest_news_updates(self, news_campus : list = None, search_bar_query: str = None,):
        if not any([search_bar_query, news_campus]):
            return "At least one parameter of this function is required. Neither Search query and news campus received. Please provide at least one parameter to perform search."
        
        search_url = "https://asu.campuslabs.com/engage/organizations"
        params = []
        news_campus_ids = { "ASU Downtown":"257211","ASU Online":"257214","ASU Polytechnic":"257212","ASU Tempe":"254417","ASU West Valley":"257213","Fraternity & Sorority Life":"257216","Housing & Residential Life":"257215"}
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif news_campus:
            doc_title = " ".join(news_campus)
        else:
            doc_title = None

        if news_campus:
            campus_id_array = [news_campus_ids[campus] for campus in news_campus if campus in news_campus_ids]
            if campus_id_array:
                params.extend([f"branches={campus_id}" for campus_id in campus_id_array])
                
        if search_bar_query:
            params.append(f"query={search_bar_query.lower().replace(' ', '%20')}")
        
        if params:
            search_url += "?" + "&".join(params)
        
        return await self.utils.perform_web_search(search_url,doc_title=doc_title, doc_category="news_info")
    
    async def get_latest_social_media_updates(self,  account_name: list, search_bar_query: str = None,):
        if not any([search_bar_query, account_name]):
            return "At least one parameter of this function is required. Neither Search query and news campus received. Please provide at least one parameter to perform search."
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif account_name:
            doc_title = " ".join(account_name)
        else:
            doc_title = None
        account_OBJECTs = {
            "@ArizonaState": [
                "https://x.com/ASU", 
                "https://www.instagram.com/arizonastate"
            ],
            "@SunDevilAthletics": [
                "https://x.com/TheSunDevils", 
            ],
            "@SparkySunDevil": [
                "https://x.com/SparkySunDevil", 
                "https://www.instagram.com/SparkySunDevil"
            ],
            "@ASUFootball": [
                "https://x.com/ASUFootball", 
                "https://www.instagram.com/sundevilfb/"
            ],
            "@ASUFootball": [
                "https://x.com/ASUFootball", 
                "https://www.instagram.com/sundevilfb/"
            ],
        }
        
        # Collect URLs for specified account names
        final_search_array = []
        for name in account_name:
            if name in account_OBJECTs:
                final_search_array.extend(account_OBJECTs[name])
        
        # If no URLs found for specified accounts, return empty list
        if not final_search_array:
            return []
        
        # Perform web search on each URL asynchronously
        search_results = []
        for url in final_search_array:
            search_result = await self.utils.perform_web_search(url,search_bar_query,doc_title=doc_title, doc_category="social_media_updates")
            search_results.extend(search_result)
        return search_results
    
    async def get_latest_sport_updates(self, search_bar_query: str = None, sport: str = None, league: str = None, match_date: str = None):
        """
        Comprehensive function to retrieve ASU sports updates using multiple data sources.
        
        Args:
            query: General search query for sports updates
            sport: Specific sport to search
            league: Sports league
            match_date: Specific match date
        
        
        Returns:
            List of sports updates or detailed information
        """
        # Validate input parameters
        if not any([search_bar_query, sport, league, match_date]):
            return "Please provide at least one parameter to perform the search."
        
        
        dynamic_query = f"ASU {sport if sport else ''} {search_bar_query if search_bar_query else ''} {league if league else ''} {match_date} site:(sundevils.com OR espn.com) -inurl:video"
        search_url=f"https://www.google.com/search?q={urllib.parse.quote(dynamic_query)}"
         
        
        return await self.utils.perform_web_search(search_url)

    async def get_library_resources(self, search_bar_query: str = None, resource_type: str = 'All Items'):
        """
        Retrieve ASU Library resources using ASU Library Search with robust parameter handling.

        Args:
            search_bar_query (str, optional): Search term for library resources.
            resource_type (str, optional): Type of resource to search. Defaults to 'All Items'.

        Returns:
            Union[str, dict]: Search results or error message.
        """
        # Comprehensive input validation with improved error handling
        if not search_bar_query:
            return "Error: Search query is required."
        
        # Use class-level constants for mappings to improve maintainability
        RESOURCE_TYPE_MAPPING = {
            'All Items': 'any', 'Books': 'books', 'Articles': 'articles', 
            'Journals': 'journals', 'Images': 'images', 'Scores': 'scores', 
            'Maps': 'maps', 'Sound recordings': 'audios', 'Video/Film': 'videos'
        }
        
        # Validate resource type and language with more graceful handling
        resource_type = resource_type if resource_type in RESOURCE_TYPE_MAPPING else 'All Items'
        
        # URL encode the search query to handle special characters
        encoded_query = urllib.parse.quote(search_bar_query)
        
        # Construct search URL with more robust parameter handling
        search_url = (
            f"https://search.lib.asu.edu/discovery/search"
            f"?query=any,contains,{encoded_query}"
            f",AND&pfilter=rtype,exact,{RESOURCE_TYPE_MAPPING[resource_type]}"
            "&tab=LibraryCatalog"
            "&search_scope=MyInstitution"
            "&vid=01ASU_INST:01ASU"
            "&lang=en"
            "&mode=advanced"
            "&offset=0"
        )
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif resource_type:
            doc_title = resource_type
        else:
            doc_title = None
        try:
            # Add error handling for web search
            return await self.utils.perform_web_search(search_url,doc_title=doc_title, doc_category ="library_resources")
        except Exception as e:
            return f"Error performing library search: {str(e)}"
        
    async def get_latest_scholarships(self, search_bar_query: str = None, academic_level:str = None,eligible_applicants: str =None, citizenship_status: str = None, gpa: str = None, focus : str = None):
    
        if not any([search_bar_query, academic_level, citizenship_status, gpa,  eligible_applicants, focus]):
            return "Please provide at least one parameter to perform the search."
        
        results =[]
        doc_title = ""
        if search_bar_query:
            doc_title = search_bar_query
        elif academic_level:
            doc_title = academic_level
        elif citizenship_status:
            doc_title = citizenship_status
        # elif college:
        #     doc_title = college
        elif focus:
            doc_title = focus
        else:
            doc_title = None
            
        
        search_url = f"https://goglobal.asu.edu/scholarship-search"
        
        query = f"academiclevel={academic_level}&citizenship_status={citizenship_status}&gpa={gpa}"
        # &college={college}
        
        result = await self.utils.perform_web_search(search_url,query, doc_title=doc_title, doc_category ="scholarships_info")
        
        
        results.append(result)
        
        
        search_url = f"https://onsa.asu.edu/scholarships"
        
        query = f"search_bar_query={search_bar_query}&citizenship_status={citizenship_status}&eligible_applicants={eligible_applicants}&focus={focus}"
        
        
        
        result = await self.utils.perform_web_search(search_url,query, doc_title=doc_title,doc_category ="scholarships_info")
        
        
        results.append(result)
        
            
        
        return results
       
    async def get_latest_job_updates( self, search_bar_query: Optional[Union[str, List[str]]] = None, job_type: Optional[Union[str, List[str]]] = None, job_location: Optional[Union[str, List[str]]] = None):
        """
        Comprehensive function to retrieve ASU Job updates using multiple data sources.
        
        Args:
            Multiple search parameters for job filtering with support for both string and list inputs
        
        Returns:
            List of search results
        """
        # Helper function to normalize input to list
        def normalize_to_list(value):
            if value is None:
                return None
            return value if isinstance(value, list) else [value]
        
        # Normalize all inputs to lists
        query_params = {
            'search_bar_query': normalize_to_list(search_bar_query),
            'job_type': normalize_to_list(job_type),
            'job_location': normalize_to_list(job_location),
        }
        
        # Remove None values
        query_params = {k: v for k, v in query_params.items() if v is not None}
        
        # Validate that at least one parameter is provided
        if not query_params:
            return "Please provide at least one parameter to perform the search."
        
        # Convert query parameters to URL query string
        # Ensure each parameter is converted to a comma-separated string if it's a list
        query_items = []
        for k, v in query_params.items():
            if isinstance(v, list):
                query_items.append(f"{k}={','.join(map(str, v))}")
            else:
                query_items.append(f"{k}={v}")
        
        query = '&'.join(query_items)
        
        search_url = "https://app.joinhandshake.com/stu/postings"
        
        results = []
        logger.info(f"Requested search query : {query}")
        doc_title = ""
        if search_bar_query:
            doc_title = " ".join(search_bar_query) if isinstance(search_bar_query, list) else search_bar_query
        elif job_type:
            doc_title = " ".join(job_type) if isinstance(job_type, list) else job_type
        elif job_location:
            doc_title = " ".join(job_location) if isinstance(job_location, list) else job_location
        else:
            doc_title = None
            
        result = await self.utils.perform_web_search(search_url, query, doc_title=doc_title, doc_category ="job_updates")
        results.append(result)
        
        return results       
    
    async def get_latest_class_information(self,search_bar_query: Optional[str] = None,class_term: Optional[str] = None,subject_name: Optional[Union[str, List[str]]] = None, 
    num_of_credit_units: Optional[Union[str, List[str]]] = None, 
    class_level: Optional[Union[str, List[str]]] = None,
    class_session: Optional[Union[str, List[str]]] = None,
    class_days: Optional[Union[str, List[str]]] = None,
    class_location: Optional[Union[str, List[str]]] = None,
    class_seat_availability : Optional[str] = None,
    ) -> str:
        """
        Optimized function to generate a search URL for ASU class catalog with flexible input handling.
        
        Args:
            Multiple optional parameters for filtering class search
        
        Returns:
            Constructed search URL for class catalog
        """
        
        # Helper function to convert input to query string
        
        
        
        DAYS_MAP = {
            'Monday': 'MON',
            'Tuesday': 'TUES', 
            'Wednesday': 'WED', 
            'Thursday': 'THURS', 
            'Friday': 'FRI', 
            'Saturday': 'SAT', 
            'Sunday': 'SUN'
        }
        
        
        CLASS_LEVEL_MAP = {
        'Lower division': 'lowerdivision',
        'Upper division': 'upperdivision', 
        'Undergraduate': 'undergrad',
        'Graduate': 'grad',
        '100-199': '100-199',
        '200-299': '200-299',
        '300-399': '300-399',
        '400-499': '400-499'
        }
        
        SESSION_MAP = {
            'A': 'A',
            'B': 'B', 
            'C': 'C',
            'Other': 'DYN'
        }
        
       

        TERM_MAP= {
            'Spring 2025': '2251',
            'Fall 2024': '2247', 
            'Summer 2024': '2244',
            'Spring 2024': '2241',
            'Fall 2023': '2237', 
            'Summer 2023': '2234'
        }
        
        CREDIT_UNITS_MAP = {
            '0': 'Less than 1',
            '1': '1',
            '2': '2',
            '3': '3',
            '4': '4',
            '5': '5',
            '6': '6',
            '7': '7 or more'
        }


        
        unmapped_items = []
        
        def _convert_to_query_string(input_value: Optional[Union[str, List[str]]], mapping: Dict[str, str]) -> str:
            global unmapped_items
            unmapped_items = []
            
            # Handle None input
            if input_value is None:
                return ''
            
            # Ensure input is a list
            if isinstance(input_value, str):
                input_value = [input_value]
            
            # Process each input value
            mapped_values = []
            for value in input_value:
                # Check if value exists in mapping
                if value in mapping:
                    mapped_values.append(mapping[value])
                else:
                    # Add unmapped items to global list
                    unmapped_items.append(value)
            
            # Join mapped values with URL-encoded comma
            return '%2C'.join(mapped_values) if mapped_values else ''
        
        
        
        search_bar_query = (search_bar_query or '') + ' ' + ' '.join(unmapped_items)
        search_bar_query+=subject_name
        search_bar_query = search_bar_query.strip().replace(" ", "%20")
        
        
        params = {
            'advanced': 'true',
            'campus': _convert_to_query_string(class_location, LOCATION_MAP),
            'campusOrOnlineSelection': 'A',
            'daysOfWeek': _convert_to_query_string(class_days, DAYS_MAP),
            'honors': 'F',
            'keywords': search_bar_query,
            'level': _convert_to_query_string(class_level, CLASS_LEVEL_MAP),
            'promod': 'F',
            'searchType': "open" if class_seat_availability == "Open" else "all",
            'session': _convert_to_query_string(class_session, SESSION_MAP),
            'term': _convert_to_query_string(class_term, TERM_MAP),
            'units': _convert_to_query_string(num_of_credit_units, CREDIT_UNITS_MAP)
        }
        
        logger.info(params)

        # Remove None values and construct URL
        search_url = 'https://catalog.apps.asu.edu/catalog/classes/classlist?' + '&'.join(
            f'{key}={value}' 
            for key, value in params.items() 
            if value is not None and value != ''
        )
        
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif subject_name:
            doc_title = " ".join(subject_name) if isinstance(subject_name, list) else subject_name
        elif class_term:
            doc_title = class_term
        elif class_level:
            doc_title = " ".join(class_level) if isinstance(class_level, list) else class_level
        elif class_location:
            doc_title = " ".join(class_location) if isinstance(class_location, list) else class_location
        elif class_session:
            doc_title = " ".join(class_session) if isinstance(class_session, list) else class_session
        elif num_of_credit_units:
            doc_title = " ".join(num_of_credit_units) if isinstance(num_of_credit_units, list) else num_of_credit_units
        elif class_days:
            doc_title = " ".join(class_days) if isinstance(class_days, list) else class_days

        elif class_seat_availability:
            doc_title = class_seat_availability
        else:
            doc_title = None

        return await self.utils.perform_web_search(search_url,doc_title=doc_title, doc_category ="classes_info")       
     
    async def notify_discord_helpers(self, short_message_to_helper: str) -> str:
        self.guild = self.discord_state.get('target_guild')
        self.user_id=self.discord_state.get('user_id')
        self.user=self.discord_state.get('user')
        logger.info(f"Initialized Discord Guild : {self.guild}")

        if not request_in_dm:
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        await self.utils.update_text("Checking available discord helpers...")

        logger.info("Contact Model: Handling contact request for helper notification")

        try:

            if not self.guild:
                return "Unable to find the server. Please try again later."

            # Check if user is already connected to a helper
            existing_channel = discord.self.utils.get(self.guild.channels, name=f"help-{self.user_id}")
            if existing_channel:
                self.utils.update_ground_sources([existing_channel.jump_url])
                return f"User already has an open help channel."

            # Find helpers
            helper_role = discord.self.utils.get(self.guild.roles, name="Helper")
            if not helper_role:
                return "Unable to find helpers. Please contact an administrator."

            helpers = [member for member in self.guild.members if helper_role in member.roles and member.status != discord.Status.offline]
            if not helpers:
                return "No helpers are currently available. Please try again later."

            # Randomly select a helper
            selected_helper = random.choice(helpers)

            # Create a private channel
            overwrites = {
                self.guild.default_role: discord.PermissionOverwrite(read_messages=False),
                user: discord.PermissionOverwrite(read_messages=True, send_messages=True),
                selected_helper: discord.PermissionOverwrite(read_messages=True, send_messages=True)
            }
            
            category = discord.self.utils.get(self.guild.categories, name="Customer Service")
            if not category:
                return "Unable to find the Customer Service category. Please contact an administrator."

            channel = await self.guild.create_text_channel(f"help-{self.user_id}", category=category, overwrites=overwrites)

            # Send messages
            await channel.send(f"{user.mention} and {selected_helper.mention}, this is your help channel.")
            await channel.send(f"User's message: {short_message_to_helper}")

            # Notify the helper via DM
            await selected_helper.send(f"You've been assigned to a new help request. Please check {channel.mention}")
            self.utils.update_ground_sources([channel.jump_url])
            return f"Server Helper Assigned: {selected_helper.name}\n"

        except Exception as e:
            logger.error(f"Error notifying helpers: {str(e)}")
            return f"An error occurred while notifying helpers: {str(e)}"

    async def notify_moderators(self, short_message_to_moderator: str) -> str:
        self.guild = self.discord_state.get('target_guild')
        self.user_id=self.discord_state.get('user_id')
        self.user=self.discord_state.get('user')
        
        logger.info(f"Initialized Discord Guild : {self.guild}")


        if not self.discord_state.get('request_in_dm'):
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        await self.utils.update_text("Checking available discord moderators...")

        logger.info("Contact Model: Handling contact request for moderator notification")

        try:
            if not self.guild:
                return "Unable to find the server. Please try again later."

            # Check if user is already connected to a helper
            existing_channel = discord.self.utils.get(self.guild.channels, name=f"support-{self.user_id}")
            if existing_channel:
                self.utils.update_ground_sources([existing_channel.jump_url])
                return f"User already has an open support channel."
            # Find helpers/moderators
            helper_role = discord.self.utils.get(self.guild.roles, name="mod")
            if not helper_role:
                return "Unable to find helpers. Please contact an administrator."

            helpers = [member for member in self.guild.members if helper_role in member.roles]
            if not helpers:
                return "No helpers are currently available. Please try again later."

            # Randomly select a helper
            selected_helper = random.choice(helpers)

            # Create a private channel
            overwrites = {
                self.guild.default_role: discord.PermissionOverwrite(read_messages=False),
                self.user: discord.PermissionOverwrite(read_messages=True, send_messages=True),
                selected_helper: discord.PermissionOverwrite(read_messages=True, send_messages=True)
            }
            
            category = discord.self.utils.get(self.guild.categories, name="Customer Service")
            if not category:
                return "Unable to find the Customer Service category. Please contact an administrator."

            channel = await self.guild.create_text_channel(f"support-{self.user_id}", category=category, overwrites=overwrites)

            # Send messages
            await channel.send(f"{self.user.mention} and {selected_helper.mention}, this is your support channel.")
            await channel.send(f"User's message: {short_message_to_moderator}")

            # Notify the helper via DM
            await selected_helper.send(f"You've been assigned to a new support request. Please check {channel.mention}")
            self.utils.update_ground_sources([channel.jump_url])
            return f"Moderator Assigned: {selected_helper.name}"

        except Exception as e:
            logger.error(f"Error notifying moderators: {str(e)}")
            return f"An error occurred while notifying moderators: {str(e)}"

    async def start_recording_discord_call(self,channel_id:Any) -> str: 

        
        logger.info(f"Initialized Discord Guild : {self.guild}")
        await self.utils.update_text("Checking user permissions...")
       
        if not self.discord_state.get('user_has_mod_role'):
            return "User does not have enough permissions to start recording a call. This command is only accessible by moderators. Exiting command..."

        if not self.discord_state.get('request_in_dm'):
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        if not self.discord_state.get('user_voice_channel_id'):
            return "User is not in a voice channel. User needs to be in a voice channel to start recording. Exiting command..."

        logger.info("Discord Model: Handling recording request")

        return f"Recording started!"

    async def create_discord_forum_post(self, title: str, category: str, body_content_1: str, body_content_2: str, body_content_3: str, link:str=None) -> str:
        self.guild = self.discord_state.get('target_guild')
        
        logger.info(f"Initialized Discord Guild : {self.guild}")
        await self.utils.update_text("Checking user permissions...")


        if not self.discord_state.get('request_in_dm'):
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        logger.info("Discord Model: Handling discord forum request with context")

        try:
            if not self.guild:
                return "Unable to find the server. Please try again later."
            try:
                
                # Find the forum channel 
                forum_channel = discord.self.utils.get(self.guild.forums, name='qna')  # Replace with your forum channel name
            except Exception as e:
                logger.error(f"Error finding forum channel: {str(e)}")
                return f"An error occurred while finding the forum channel: {str(e)}"
            if not forum_channel:
                return "Forum channel not found. Please ensure the forum exists."

            # Create the forum post
            content = f"{body_content_1}\n\n{body_content_2}\n\n{body_content_3}".strip()
            if link:
                content+=f"\n[Link]({link})"
            try:
                logger.info(f"Forum channel ID: {forum_channel.id if forum_channel else 'None'}")
                
                thread = await forum_channel.create_thread(
                    name=title,
                    content=content,
                )

            except Exception as e:
                
                logger.error(f"Error creating forum thread: {str(e)}")
                return f"An error occurred while creating the forum thread: {str(e)}"
            logger.info(f"Created forum thread {thread.message.id} {type(thread)}")
            
            self.utils.update_ground_sources([f"https://discord.com/channels/1256076931166769152/{thread.id}"])
            return f"Forum post created successfully.\nTitle: {title}\nDescription: {content[:100]}...\n"
        

        except discord.errors.Forbidden:
            return "The bot doesn't have permission to create forum posts. Please contact an administrator."
        except discord.errors.HTTPException as e:
            logger.error(f"HTTP error creating forum post: {str(e)}")
            return f"An error occurred while creating the forum post: {str(e)}"
        except Exception as e:
            logger.error(f"Error creating forum post: {str(e)}")
            return f"An unexpected error occurred while creating the forum post: {str(e)}"
    
    async def create_discord_announcement(self, ping: str, title: str, category: str, body_content_1: str, body_content_2: str, body_content_3: str, link:str = None) -> str:
        self.discord_client = self.discord_state.get('discord_client')
        self.guild = self.discord_state.get('target_guild')
        
        await self.utils.update_text("Checking user permissions...")


        logger.info(f"Discord Model: Handling discord announcement request with context")

        if not self.discord_state.get('request_in_dm'):
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        if not self.discord_state.get('user_has_mod_role'):
            return "User does not have enough permissions to create an announcement. This command is only accessible by moderators. Exiting command..."

        try:
            # Find the announcements channel
            announcements_channel = discord.self.utils.get(self.discord_client.get_all_channels(), name='announcements')
            if not announcements_channel:
                return "Announcements channel not found. Please ensure the channel exists."

            # Create the embed
            embed = discord.Embed(title=title, color=discord.Color.blue())
            embed.add_field(name="Category", value=category, inline=False)
            embed.add_field(name="Details", value=body_content_1, inline=False)
            if body_content_2:
                embed.add_field(name="Additional Information", value=body_content_2, inline=False)
            if body_content_3:
                embed.add_field(name="More Details", value=body_content_3, inline=False)
            if link:
                embed.add_field(name="Links", value=link, inline=False)

            # Send the announcement
            message = await announcements_channel.send(content="@som", embed=embed)
            self.utils.update_ground_sources([message.jump_url])
            return f"Announcement created successfully."

        except Exception as e:
            logger.error(f"Error creating announcement: {str(e)}")
            return f"An error occurred while creating the announcement: {str(e)}"
  
    async def create_discord_event(self, title: str, time_start: str, time_end: str, description: str, img_provided: Any = None) -> str:
        self.guild = self.discord_state.get('target_guild')
        
        logger.info(f"Initialized Discord Guild : {self.guild}")
        await self.utils.update_text("Checking user permissions...")


        if not self.discord_state.get('request_in_dm'):
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        if not self.discord_state.get('user_has_mod_role'):
            return "User does not have enough permissions to create an event. This command is only accessible by moderators. Exiting command..."

        logger.info("Discord Model: Handling discord event creation request")

        try:
            if self.guild:
                return "Unable to find the server. Please try again later."

            # Parse start and end times
            start_time = datetime.fromisoformat(time_start)
            end_time = datetime.fromisoformat(time_end)

            # Create the event
            event = await self.guild.create_scheduled_event(
                name=title,
                description=description,
                start_time=start_time,
                end_time=end_time,
                location="Discord",  # or specify a different location if needed
                privacy_level=discord.PrivacyLevel.guild_only
            )

            # If an image was provided, set it as the event cover
            if img_provided:
                await event.edit(image=img_provided)

            # Create an embed for the event announcement
            embed = discord.Embed(title=title, description=description, color=discord.Color.blue())
            embed.add_field(name="Start Time", value=start_time.strftime("%Y-%m-%d %H:%M:%S"), inline=True)
            embed.add_field(name="End Time", value=end_time.strftime("%Y-%m-%d %H:%M:%S"), inline=True)
            embed.add_field(name="Location", value="Discord", inline=False)
            embed.set_footer(text=f"Event ID: {event.id}")

            # Send the announcement to the announcements channel
            announcements_channel = discord.self.utils.get(self.guild.text_channels, name="announcements")
            if announcements_channel:
                await announcements_channel.send(embed=embed)
            
            self.utils.update_ground_sources([event.url])

            return f"Event created successfully.\nTitle: {title}\nDescription: {description[:100]}...\nStart Time: {start_time}\nEnd Time: {end_time}\n"

        except discord.errors.Forbidden:
            return "The bot doesn't have permission to create events. Please contact an administrator."
        except ValueError as e:
            return f"Invalid date format: {str(e)}"
        except Exception as e:
            logger.error(f"Error creating event: {str(e)}")
            return f"An unexpected error occurred while creating the event: {str(e)}"
    
    async def search_discord(self,query:str):
        results = await self.utils.perform_web_search(optional_query=query,doc_title =query)
        return results
    
    async def create_discord_poll(self, question: str, options: List[str], channel_name: str) -> str:
        self.guild = self.discord_state.get('target_guild')
        

        await self.utils.update_text("Checking user permissions...")

        if not self.discord_state.get('request_in_dm'):
            return "User can only access this command in private messages. Exiting command."

        if not self.discord_state.get('user_has_mod_role'):
            return "User does not have enough permissions to create a poll. This command is only accessible by moderators. Exiting command..."

        logger.info("Discord Model: Handling discord poll creation request")

        try:
            if not self.guild:
                return "Unable to find the server. Please try again later."

            # Find the specified channel
            channel = discord.self.utils.get(self.guild.text_channels, name=channel_name)
            if not channel:
                return f"Channel '{channel_name}' not found. Please check the channel name and try again."

            # Create the poll message
            poll_message = f"📊 **{question}**\n\n"
            emoji_options = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
            try:
                for i, option in enumerate(options):  # Limit to 10 options
                    poll_message += f"{emoji_options[i]} {option}\n"
                    
            except Exception as e:
                logger.error(f"Error creating poll options: {str(e)}")
                return f"An unexpected error occurred while creating poll options: {str(e)}"
            
            # Send the poll message
            try:
                poll = await channel.send(poll_message)
            except Exception as e:
                logger.error(f"Error sending poll message: {str(e)}")
                return  f"An unexpected error occurred while sending poll: {str(e)}"
            
            self.utils.update_ground_sources([poll.jump_url])  

            # Add reactions
            try:
                
                for i in range(len(options)):
                    await poll.add_reaction(emoji_options[i])
            except Exception as e:
                logger.error(f"Error adding reactions to poll: {str(e)}")
                return f"An unexpected error occurred while adding reactions to poll: {str(e)}"
            
            return f"Poll created successfully in channel '{channel_name}'.\nQuestion: {question}\nOptions: {', '.join(options)}"

        except discord.errors.Forbidden:
            return "The bot doesn't have permission to create polls or send messages in the specified channel. Please contact an administrator."
        except Exception as e:
            logger.error(f"Error creating poll: {str(e)}")
            return f"An unexpected error occurred while creating the poll: {str(e)}"
            
    def get_final_url(self,url):
        try:
            response = requests.get(url, allow_redirects=True)
            return response.url
        except Exception as e:
            logger.error(e)
            return e  

    async def access_rag_search_agent(self, instruction_to_agent: str, special_instructions: str):
        logger.info(f"Action Model : accessing search agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        try:
            response = await self.rag_rag_search_agent.determine_action(instruction_to_agent,special_instructions)
            return response
        except Exception as e:
            logger.error(f"Error in access search agent: {str(e)}")
            return f"Search Agent Not Responsive"
         
    async def access_discord_agent(self, instruction_to_agent: str,special_instructions: str):
        logger.info(f"Action Model : accessing discord agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        try:
            response = await self.discord_agent.determine_action(instruction_to_agent,special_instructions)
            
            return response
        except Exception as e:
            logger.error(f"Error in access discord agent: {str(e)}")
            return f"Discord Agent Not Responsive"
        
    async def get_user_profile_details(self) -> str:
        """Retrieve user profile details from the Discord server"""
        self.guild = self.discord_state.get('target_guild')
        self.user_id = self.discord_state.get('user_id')
        logger.info(f"Discord Model: Handling user profile details request for user ID: {user_id}")

        if not request_in_dm:
            return "User can only access this command in private messages. It seems like the user is trying to access this command in a discord server. Exiting command."

        try:
            # If no user_id is provided, use the requester's ID
            if not user_id:
                user_id = self.user_id

            member = await self.guild.fetch_member(user_id)
            if not member:
                return f"Unable to find user with ID {user_id} in the server."

            # Fetch user-specific data (customize based on your server's setup)
            join_date = member.joined_at.strftime("%Y-%m-%d")
            roles = [role.name for role in member.roles if role.name != "@everyone"]
            
            # You might need to implement these functions based on your server's systems
            # activity_points = await self.get_user_activity_points(user_id)
            # leaderboard_position = await self.get_user_leaderboard_position(user_id)
            # - Activity Points: {activity_points}
            # - Leaderboard Position: {leaderboard_position}

            profile_info = f"""
            User Profile for {member.name}#{member.discriminator}:
            - Join Date: {join_date}
            - Roles: {', '.join(roles)}
            - Server Nickname: {member.nick if member.nick else 'None'}
            """

            return profile_info.strip()

        except discord.errors.NotFound:
            return f"User with ID {user_id} not found in the server."
        except Exception as e:
            logger.error(f"Error retrieving user profile: {str(e)}")
            return f"An error occurred while retrieving the user profile: {str(e)}"
    
    async def get_discord_server_info(self) -> str:
             
        self.discord_client = self.discord_state.get('discord_client')
        logger.info(f"Initialized Discord Client : {self.discord_client}")
        self.guild = self.discord_state.get("target_guild")
        
        logger.info(f"Initialized Discord Guild : {self.guild}")
        """Create discord forum post callable by model"""

        
        logger.info(f"Discord Model : Handling discord server info request with context")
                
        
        return f"""1.Sparky Discord Server - Sparky Discord Server is a place where ASU Alumni's or current students join to hangout together, have fun and learn things about ASU together and quite frankly!
        2. Sparky Discord Bot -  AI Agent built to help people with their questions regarding ASU related information and sparky's discord server. THis AI Agent can also perform discord actions for users upon request."""
    
    async def access_live_status_agent(self, instruction_to_agent: str, special_instructions: str):
        logger.info(f"Action Model : accessing live status agent with instruction {instruction_to_agent} with special instructions {special_instructions}")
        
        try:
            response = await self.live_status_agent.determine_action(instruction_to_agent,special_instructions)
            return response
        except Exception as e:
            logger.error(f"Error in deep search agent: {str(e)}")
            return "I apologize, but I couldn't retrieve the information at this time."
              
    async def send_bot_feedback(self, feedback: str) -> str:
        self.user = self.discord_state.get('user') 
        self.discord_client = self.discord_state.get('discord_client')
        
        await self.utils.update_text("Opening feedbacks...")
        
        logger.info("Contact Model: Handling contact request for server feedback")

        try:
            # Find the feedbacks channel
            feedbacks_channel = discord.self.utils.get(self.discord_client.get_all_channels(), name='feedback')
            if not feedbacks_channel:
                return "feedbacks channel not found. Please ensure the channel exists."

            # Create an embed for the feedback
            embed = discord.Embed(title="New Server feedback", color=discord.Color.green())
            embed.add_field(name="feedback", value=feedback, inline=False)
            embed.set_footer(text=f"Suggested by {self.user.name}")

            # Send the feedback to the channel
            message = await feedbacks_channel.send(embed=embed)

            # Add reactions for voting
            await message.add_reaction('👍')
            await message.add_reaction('👎')
            
            self.utils.update_ground_sources([message.jump_url])
            
            return f"Your feedback has been successfully submitted."

        except Exception as e:
            logger.error(f"Error sending feedback: {str(e)}")
            return f"An error occurred while sending your feedback: {str(e)}"
    
    def _get_chat_history(self, user_id):
        return self.conversations.get(user_id, [])

    def _save_message(self, user_id: str, role: str, content: str):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({
            "role": role,
            "parts": [{"text": content}]
        })
        
        # Limit the conversation length to 3 messages per user
        if len(self.conversations[user_id]) > 3:
            self.conversations[user_id].pop(0)

    async def access_google_agent(self, original_query: str, detailed_query: str, generalized_query: str, relative_query: str, categories: list):
        self.firestore.update_message("category", categories)
        
        user_id = self.discord_state.get('user_id')
        responses=[]
        logger.info(f"Action Model: accessing Google Search with instruction {original_query}")
        try:
            # Perform database search
            queries = [
                {"search_bar_query": original_query},
                {"search_bar_query": detailed_query},
                {"search_bar_query": generalized_query},
                {"search_bar_query": relative_query}
            ]
            for query in queries:
                response = await self.utils.perform_database_search(query["search_bar_query"], categories) or []
                responses.append(response)

            responses = [resp for resp in responses if resp]
        except:
            logger.error("No results found in database")
            pass
        # Get chat history
        
        chat_history = self._get_chat_history(user_id)

        # Prepare the prompt
        prompt = f"""
        
        {self.app_config.get_google_agent_prompt()}
        
        - If applicable, you may use the related database information : {responses}
        
        Chat History:
        {chat_history}

        User's Query: {original_query}

        Deliver a direct, actionable response that precisely matches the query's specificity."""
        
        try:     
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=GenerateContentConfig(
                    tools=[self.google_search_tool],
                    response_modalities=["TEXT"],
                    system_instruction=f"{self.app_config.get_google_agent_instruction()}",
                    max_output_tokens=600
                )
            )
            
            grounding_sources = [self.get_final_url(chunk.web.uri) for candidate in response.candidates if candidate.grounding_metadata and candidate.grounding_metadata.grounding_chunks for chunk in candidate.grounding_metadata.grounding_chunks if chunk.web]
            
            self.utils.update_ground_sources(grounding_sources)
            
            response_text = "".join([part.text for part in response.candidates[0].content.parts if part.text])


            # Save the interaction to chat history
            self._save_message(user_id, "user", original_query)
            self._save_message(user_id, "model", response_text)

            logger.info(response_text)

            if not response_text:
                logger.error("No response from Google Search")
                return None
            return response_text
        except Exception as e:
            logger.info(f"Google Search Exception {e}")
            return responses 
        

        self.firestore.update_message("category", categories)
        
        user_id = self.discord_state.get('user_id')
        responses=[]
        logger.info(f"Action Model: accessing Google Search with instruction {original_query}")
        try:
            # Perform database search
            queries = [
                {"search_bar_query": original_query},
                {"search_bar_query": detailed_query},
                {"search_bar_query": generalized_query},
                {"search_bar_query": relative_query}
            ]
            for query in queries:
                response = await self.utils.perform_database_search(query["search_bar_query"], categories) or []
                responses.append(response)

            responses = [resp for resp in responses if resp]
        except:
            logger.error("No results found in database")
            pass
        # Get chat history
        
        chat_history = self._get_chat_history(user_id)

        # Prepare the prompt
        prompt = f"""
        
        {self.app_config.get_google_agent_prompt()}
        
        - If applicable, you may use the related database information : {responses}
        
        Chat History:
        {chat_history}

        User's Query: {original_query}

        Deliver a direct, actionable response that precisely matches the query's specificity."""
        
        try:     
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=GenerateContentConfig(
                    tools=[self.google_search_tool],
                    response_modalities=["TEXT"],
                    system_instruction=f"{self.app_config.get_google_agent_instruction()}",
                    max_output_tokens=600
                )
            )
            
            grounding_sources = [self.get_final_url(chunk.web.uri) for candidate in response.candidates if candidate.grounding_metadata and candidate.grounding_metadata.grounding_chunks for chunk in candidate.grounding_metadata.grounding_chunks if chunk.web]
            
            self.utils.update_ground_sources(grounding_sources)
            
            response_text = "".join([part.text for part in response.candidates[0].content.parts if part.text])


            # Save the interaction to chat history
            self._save_message(user_id, "user", original_query)
            self._save_message(user_id, "model", response_text)

            logger.info(response_text)

            if not response_text:
                logger.error("No response from Google Search")
                return None
            return response_text
        except Exception as e:
            logger.info(f"Google Search Exception {e}")
            return responses 