from utils.common_imports import *

class Library_Agent_Tools:
    def __init__(self,firestore,utils,logger):
        self.firestore = firestore
        self.utils = utils
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3
        
    async def get_live_library_status(self, status_type : list = None, date : str = None, library_names: list = None):
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
                transformed_date = datetime.strptime(date, '%b %d').strftime('2025-%m-%d')
                for library in library_names:
                    query= library_map[library]
                    search_url = f"https://asu.libcal.com/r/accessible/availability?lid={library_map[library]}&gid={gid_map[library_map[library]]}&zone=0&space=0&capacity=2&date={transformed_date}"
                    result+=await self.utils.perform_web_search(search_url, query,doc_title=doc_title, doc_category ="libraries_status")
            if not result:
                return "No rooms available currently at this specific library."
            return result
            
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
        