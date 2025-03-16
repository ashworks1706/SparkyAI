from utils.common_imports import *

class Library_Agent_Tools:
    def __init__(self,firestore,utils,logger):
        self.firestore = firestore
        self.utils = utils
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3
        

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
        