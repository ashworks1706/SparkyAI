from utils.common_imports import *

class Live_Status_Agent_Tools:
    def __init__(self, firestore,utils,logger):
        self.firestore = firestore
        self.utils = utils
        self.logger = logger
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3
                
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

        self.logger.info(shuttle_route)
        
        if len(shuttle_route) == 1:
            self.logger.info("\nOnly one route")
            route = next(iter(shuttle_route))
            return await self.utils.perform_web_search(search_url, optional_query=route,doc_title=doc_title, doc_category ="shuttles_status")

        # Multiple routes handling
        result = ""
        try:
            for route in shuttle_route:
                result += await self.utils.perform_web_search(search_url, optional_query=route,doc_title=doc_title, doc_category ="shuttles_status")
            self.logger.info("\nDone")
            return result
        except Exception as e:
            return f"Error performing shuttle search: {str(e)}"