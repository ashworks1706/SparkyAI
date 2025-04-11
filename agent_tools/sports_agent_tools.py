from utils.common_imports import *

class Sports_Agent_Tools:
    def __init__(self,firestore,utils,logger):
        self.firestore = firestore
        self.utils = utils
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3
        

    
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
    
    async def get_ticketing_info(self, search_bar_query: str = None, sport: str = None, match_date: str = None):
        if not any([search_bar_query, sport]):
            return "Please provide at least one sport to perform the search."
        
        url = "sundevils.com/tickets"
        return await self.utils.asu_scraper.scrape_content(url=url, query_type=sport, selenium=True, optional_query=sport)
        

   