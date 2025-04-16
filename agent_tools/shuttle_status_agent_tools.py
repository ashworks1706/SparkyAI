from utils.common_imports import *

class Shuttle_Status_Agent_Tools:
    def __init__(self, middleware,utils,logger):
        self.middleware = middleware
        self.utils = utils
        self.logger = logger
        self.visited_urls = set() 
        self.max_depth = 2
        self.max_links_per_page = 3
                
  
    async def get_live_shuttle_status(self, shuttle_route: list = None):
        if not shuttle_route:
            return "Error: At least one route is required"
        
        shuttle_route = set(shuttle_route)
        
        doc_title = " ".join(shuttle_route)
        search_url="https://asu-shuttles.rider.peaktransit.com/"

        self.logger.info(shuttle_route)
        
        if len(shuttle_route) == 1:
            self.logger.info("@shuttle_agent_tools.py  Only one route")
            route = next(iter(shuttle_route))
            return await self.utils.perform_web_search(search_url, optional_query=route,doc_title=doc_title, doc_category ="shuttles_status")

        # Multiple routes handling
        result = ""
        try:
            for route in shuttle_route:
                result += await self.utils.perform_web_search(search_url, optional_query=route,doc_title=doc_title, doc_category ="shuttles_status")
            self.logger.info("@shuttle_agent_tools.py  Done")
            return result
        except Exception as e:
            return f"Error performing shuttle search: {str(e)}"