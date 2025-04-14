from utils.common_imports import *

class Study_Rooms:
    def __init__(self):
        self.app_config = AppConfig()  
        self.logger = logging.getLogger(__name__)
        self.asu_scraper = ASUWebScraper({}, Utils(vector_store_class=None, asu_data_processor=None, asu_scraper=None, logger=self.logger, group_chat=[]), self.logger)
        self.temp_docs = []
        self.library_docs = []
        
    async def get_live_library_status(self, status_type: list = None, date: str = None, library_names: list = None):
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
            return "Error: At least one parameter required"
        
        library_map = {
            "Tempe Campus - Hayden Library": "13858",
            "Tempe Campus - Noble Library": "1702",
            "Downtown Phoenix Campus - Fletcher Library": "1703",
            "West Campus - Library": "1707",
            "Polytechnic Campus - Library": "1704"
        }
        
        gid_map = {
            "13858": "28619",
            "1702": "2897",
            "1703": "2898", 
            "1707": "28611",
            "1704": "2899"
        }
        
        self.temp_docs = []
        self.library_docs = []
        doc_title = " ".join(library_names) if library_names else "ASU Libraries"
        
        try:
            if "Availability" in status_type:
                search_url = "https://lib.asu.edu/hours"
                for library in library_names:
                    query = f"library={library}&date={date}"
                    self.temp_docs = await self.asu_scraper.engine_search(search_url, query)
                    self.library_docs.append({
                        "documents": self.temp_docs,
                        "search_context": f"ASU Library hours for {library} on {date}",
                        "title": f"{library} Hours"
                    })
            
            if "StudyRoomsAvailability" in status_type and date:
                transformed_date = datetime.strptime(date, '%b %d').strftime('2025-%m-%d')
                for library in library_names:
                    if library in library_map:
                        lid = library_map[library]
                        gid = gid_map[lid]
                        search_url = f"https://asu.libcal.com/r/accessible/availability?lid={lid}&gid={gid}&zone=0&space=0&capacity=2&date={transformed_date}"
                        self.temp_docs = await self.asu_scraper.engine_search(search_url, f"study rooms {library}")
                        self.library_docs.append({
                            "documents": self.temp_docs,
                            "search_context": f"Study room availability for {library} on {date}",
                            "title": f"{library} Study Rooms"
                        })
            
            if not self.library_docs:
                return "No information available for the specified libraries or date."
            
            return self.library_docs
            
        except Exception as e:
            return f"Error performing library search: {str(e)}"
    
    async def perform_web_search(self):
        # Test with sample parameters
        library_names = ["Tempe Campus - Hayden Library", "Tempe Campus - Noble Library"]
        status_types = ["Availability", "StudyRoomsAvailability"]
        date = "Dec 04"
        return await self.get_live_library_status(status_types, date, library_names)