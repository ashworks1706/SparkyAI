from utils.common_imports import *

class Library_Catalog:
    def __init__(self):
        self.app_config = AppConfig()
        self.logger = logging.getLogger(__name__)
        self.scraper = ASUWebScraper({}, Utils(vector_store_class=None, asu_data_processor=None, asu_scraper=None, logger=self.logger, group_chat=[]), self.logger)
        self.temp_docs = []
        self.library_docs = []

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
        
        doc_title = ""
        if search_bar_query:
            doc_title = search_bar_query
        elif resource_type:
            doc_title = resource_type
        else:
            doc_title = None
            
        try:
            # Add error handling for web search
            return await self.scraper.engine_search(search_url)
        except Exception as e:
            return f"Error performing library search: {str(e)}"
    
    async def perform_web_search(self):
        """
        Comprehensive background scraping for library resources from ASU Library
        to gather a wide range of academic materials.
        """
        results = []
        
        # Common academic subjects
        academic_subjects = [
            "computer science", "engineering", "psychology", "mathematics",
            "biology", "history", "economics", "physics", "chemistry", "art"
        ]
        
        # Common resource types
        resource_types = [
            "All Items", "Books", "Articles", "Journals", "Images"
        ]
        
        # Common search terms for academic resources
        search_terms = [
            "artificial intelligence", "climate change", "sustainability",
            "machine learning", "social media", "renewable energy",
            "public health", "digital humanities", "cybersecurity"
        ]
        
        # 1. Search by academic subjects
        for subject in academic_subjects[:5]:  # Limit to first 5 to avoid too many requests
            result = await self.get_library_catalog(search_bar_query=subject)
            if result and not isinstance(result, str):
                self.library_docs.append({
                    "documents": [result],
                    "search_context": f"Library resources for subject: {subject}",
                    "title": subject,
                })
        
        # 2. Search by resource types
        for resource in resource_types:
            # Use a general academic term with each resource type
            result = await self.get_library_catalog(search_bar_query="research methodology", resource_type=resource)
            if result and not isinstance(result, str):
                self.library_docs.append({
                    "documents": [result],
                    "search_context": f"Library {resource.lower()} about research methodology",
                    "title": f"Research Methodology - {resource}",
                })
        
        # 3. Search for trending academic topics
        for term in search_terms[:5]:  # Limit to first 5
            result = await self.get_library_catalog(search_bar_query=term)
            if result and not isinstance(result, str):
                self.library_docs.append({
                    "documents": [result],
                    "search_context": f"Library resources on trending topic: {term}",
                    "title": term,
                })
        
        # 4. Search for specific combinations of subject and resource type
        combinations = [
            ("data science", "Books"),
            ("virtual reality", "Articles"),
            ("quantum computing", "Journals"),
            ("educational technology", "All Items"),
            ("genetic engineering", "Articles")
        ]
        
        for query, res_type in combinations:
            result = await self.get_library_catalog(search_bar_query=query, resource_type=res_type)
            if result and not isinstance(result, str):
                self.library_docs.append({
                    "documents": [result],
                    "search_context": f"{res_type} about {query}",
                    "title": f"{query} - {res_type}",
                })
        
        # 5. Search for ASU-specific resources
        asu_terms = [
            "Arizona State University research", 
            "ASU innovation", 
            "Sun Devil scholarship"
        ]
        
        for term in asu_terms:
            result = await self.get_library_catalog(search_bar_query=term)
            if result and not isinstance(result, str):
                self.library_docs.append({
                    "documents": [result],
                    "search_context": f"ASU-specific resources: {term}",
                    "title": term,
                })
                
        return self.library_docs