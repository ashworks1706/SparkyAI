from utils.common_imports import *

class News:
    def __init__(self):
        self.app_config = AppConfig()  
        self.logger = logging.getLogger(__name__)
        self.asu_scraper = ASUWebScraper({}, Utils(vector_store_class=None, asu_data_processor=None, asu_scraper=None, logger=self.logger, group_chat=[]), self.logger)
        self.temp_docs = []
        self.news_docs = []
        self.news_campus_ids = {
            "ASU Downtown": "257211",
            "ASU Online": "257214",
            "ASU Polytechnic": "257212",
            "ASU Tempe": "254417",
            "ASU West Valley": "257213",
            "Fraternity & Sorority Life": "257216",
            "Housing & Residential Life": "257215"
        }

    async def get_latest_news_updates(self, news_campus : list = None, search_bar_query: str = None):
        if not any([search_bar_query, news_campus]):
            return "At least one parameter of this function is required. Neither Search query and news campus received. Please provide at least one parameter to perform search."
        
        search_url = "https://asu.campuslabs.com/engage/news"
        params = []
        doc_title = ""
        
        if search_bar_query:
            doc_title = search_bar_query
        elif news_campus:
            doc_title = " ".join(news_campus)
        else:
            doc_title = None

        if news_campus:
            campus_id_array = [self.news_campus_ids[campus] for campus in news_campus if campus in self.news_campus_ids]
            if campus_id_array:
                params.extend([f"branches={campus_id}" for campus_id in campus_id_array])
                
        if search_bar_query:
            params.append(f"query={search_bar_query.lower().replace(' ', '%20')}")
        
        if params:
            search_url += "?" + "&".join(params)
        
        try:
            self.temp_docs = await self.asu_scraper.engine_search(search_url, doc_title)
            self.news_docs.append({
                "documents": self.temp_docs,
                "search_context": f"ASU News information for {doc_title}",
                "title": doc_title
            })
            
            return self.news_docs
        except Exception as e:
            return f"Error performing news search: {str(e)}"
    
    async def perform_web_search(self):
        """
        Comprehensive background scraping for news that tries multiple combinations 
        of search terms and campuses to gather maximum news information.
        """
        results = []
        
        # Common search terms for news
        search_terms = ["research", "faculty", "student achievement", "grants", 
                      "athletics", "innovation", "community", "technology", "campus events", "awards"]
        
        # Get all campuses
        all_campuses = list(self.news_campus_ids.keys())
        
        # 1. Fetch by individual search terms
        for term in search_terms:
            results.append(await self.get_latest_news_updates(search_bar_query=term))
        
        # 2. Fetch by individual campuses
        for campus in all_campuses:
            results.append(await self.get_latest_news_updates(news_campus=[campus]))
        
        # 3. Selected combinations of search terms and campuses
        combinations = [
            ("research", ["ASU Tempe"]),
            ("student achievement", ["ASU Polytechnic"]),
            ("innovation", ["ASU Downtown"]),
            ("community", ["ASU West Valley"]),
            ("technology", ["ASU Online"])
        ]
        
        for term, campus in combinations:
            results.append(await self.get_latest_news_updates(
                search_bar_query=term,
                news_campus=campus
            ))
            
        # 4. Multiple campus combinations
        multi_campus_combinations = [
            (["ASU Tempe", "ASU Downtown"]),
            (["ASU Polytechnic", "ASU West Valley"]),
            (["ASU Downtown", "ASU Online"]),
            (["ASU Tempe", "ASU Polytechnic", "ASU West Valley"])
        ]
        
        for campuses in multi_campus_combinations:
            results.append(await self.get_latest_news_updates(news_campus=campuses))
            
        # 5. Multiple search terms with multiple campuses
        complex_combinations = [
            ("faculty research", ["ASU Tempe", "ASU Downtown"]),
            ("student innovation", ["ASU Polytechnic", "ASU Online"]),
            ("campus events", ["ASU West Valley", "ASU Tempe"])
        ]
        
        for term, campuses in complex_combinations:
            results.append(await self.get_latest_news_updates(
                search_bar_query=term,
                news_campus=campuses
            ))
            
        return self.news_docs