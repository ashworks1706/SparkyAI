from utils.common_imports import *

class Student_Jobs_Agent_Tools:
    def __init__(self,firestore,utils,logger):
        self.firestore = firestore
        self.utils = utils
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3
        
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
        self.logger.info("@student_jobs_agent_tools.py Requested search query : {query}")
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
    
