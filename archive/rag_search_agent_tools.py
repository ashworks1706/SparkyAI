from utils.common_imports import *

class Rag_Search_Agent_Tools:
    def __init__(self,firestore,utils, app_config,logger):
        self.firestore = firestore
        self.utils = utils
        self.app_config = app_config
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3
        
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

    async def get_library_catalog(self, search_bar_query: str = None, resource_type: str = 'All Items'):
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
            return await self.utils.perform_web_search(search_url,doc_title=doc_title, doc_category ="library_catalog")
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
       
    async def get_latest_student_jobs( self, search_bar_query: Optional[Union[str, List[str]]] = None, job_type: Optional[Union[str, List[str]]] = None, job_location: Optional[Union[str, List[str]]] = None):
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
        self.logger.info(f"Requested search query : {query}")
        doc_title = ""
        if search_bar_query:
            doc_title = " ".join(search_bar_query) if isinstance(search_bar_query, list) else search_bar_query
        elif job_type:
            doc_title = " ".join(job_type) if isinstance(job_type, list) else job_type
        elif job_location:
            doc_title = " ".join(job_location) if isinstance(job_location, list) else job_location
        else:
            doc_title = None
            
        result = await self.utils.perform_web_search(search_url, query, doc_title=doc_title, doc_category ="student_jobs")
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
        
        self.logger.info(params)

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

        return await self.utils.perform_web_search(search_url,doc_title=doc_title, doc_category ="courses_catalog")