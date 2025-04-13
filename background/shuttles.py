from utils.common_imports import *


class Shuttles:
    def __init__(self, vector_store):
        self.app_config = AppConfig()  
        self.logger = logging.getLogger(__name__)
        self.asu_scraper = ASUWebScraper({}, Utils(vector_store_class=vector_store, asu_data_processor=None, asu_scraper=None, logger=self.logger,group_chat=[]), self.logger)
        self.temp_docs = []
        self.shuttle_docs=[]
    async def get_live_shuttle_status(self, shuttle_route: list = None):
        #  self..._docs = [
        #     {
        #         "documents": [
        #             {
        #                 "id": "123",
        #                 "title": "Shuttle Schedule Update",
        #                 "content": "The shuttle schedule has been updated for the fall semester.",
        #                 "url": "https://example.com/shuttle-update",
        #                 "timestamp": "2023-10-01T10:00:00Z"
        #             },
        #             {
        #                 "id": "124",
        #                 "title": "New Shuttle Route",
        #                 "content": "A new shuttle route has been added to serve the west campus.",
        #                 "url": "https://example.com/new-shuttle-route",
        #                 "timestamp": "2023-10-02T12:00:00Z"
        #             }
        #         ],
        #         "search_context": "shuttle information",
        #         "title": "Shuttle Updates"
        #     }
        # ]
        shuttle_route = set(shuttle_route)
        
        doc_title = " ".join(shuttle_route)
        search_url="https://asu-shuttles.rider.peaktransit.com/"

        # Multiple routes handling
        self.temp_docs = []
        try:
            for route in shuttle_route:
                self.temp_docs = await self.asu_scraper.engine_search(search_url, route)
                self.shuttle_docs.append( {
                    "documents": self.temp_docs,
                    "search_context": f"ASU Shuttle live status for {route}",
                    "title": doc_title
                })
            
            return self.shuttle_docs
        except Exception as e:
            return f"Error performing shuttle search: {str(e)}"
    
    
    async def perform_web_search(self):
        # try all possible function parameters
        shuttle_routes = [ "Mercado", "Polytechnic-Tempe","Tempe-Downtown Phoenix-West","Tempe-West Express"]
        return await self.get_live_shuttle_status(shuttle_routes)
        