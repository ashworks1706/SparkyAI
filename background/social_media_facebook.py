from utils.common_imports import *

class Social_Media_Facebook:
    def __init__(self):
        self.app_config = AppConfig()
        self.logger = logging.getLogger(__name__)
        self.scraper = ASUWebScraper({}, Utils(vector_store_class=None, asu_data_processor=None, asu_scraper=None, logger=self.logger, group_chat=[]), self.logger)
        self.temp_docs = []
        self.social_media_docs = []
        self.account_OBJECTs = {
            "@ArizonaState": [
                "https://www.facebook.com/arizonastate/",
                "https://www.facebook.com/ASUColleges/"
            ],
            "@SunDevilAthletics": [
                "https://www.facebook.com/sundevilathletics/",
            ],
            "@SparkySunDevil": [
                "https://www.facebook.com/SparkysASU/",
            ],
            "@ASUFootball": [
                "https://www.facebook.com/ASUFootball/",
            ],
            "@ASUAlumni": [
                "https://www.facebook.com/ASUAlumni/",
            ],
            "@ASUEngineering": [
                "https://www.facebook.com/ASUEngineering/",
            ],
        }

    async def get_latest_social_media_updates(self, account_name: list, search_bar_query: Optional[str] = None):
        if not any([search_bar_query, account_name]):
            return "At least one parameter of this function is required. Neither search query nor account name received. Please provide at least one parameter to perform search."
        
        doc_title = ""
        if search_bar_query:
            doc_title = search_bar_query
        elif account_name:
            doc_title = " ".join(account_name)
        else:
            doc_title = None
            
        # Collect URLs for specified account names
        final_search_array = []
        for name in account_name:
            if name in self.account_OBJECTs:
                final_search_array.extend(self.account_OBJECTs[name])
        
        # If no URLs found for specified accounts, return empty list
        if not final_search_array:
            return []
        
        # Perform web search on each URL
        search_results = []
        for url in final_search_array:
            try:
                search_url = url
                if search_bar_query:
                    # Add search query to URL if provided
                    search_url = f"{url}?q={search_bar_query.lower().replace(' ', '%20')}"
                
                result = await self.scraper.engine_search(search_url, doc_title)
                self.temp_docs = result
                self.social_media_docs.append({
                    "documents": self.temp_docs,
                    "search_context": f"Facebook updates from {url} for {doc_title}",
                    "title": doc_title,
                })
                search_results.append(self.social_media_docs[-1])
            except Exception as e:
                self.logger.error(f"Error fetching Facebook data: {str(e)}")
        
        return search_results
    
    async def perform_web_search(self):
        """
        Comprehensive background scraping for Facebook updates from multiple accounts
        to gather maximum social media information.
        """
        results = []
        
        # Common ASU-related accounts
        accounts_to_search = [
            ["@ArizonaState"],
            ["@SunDevilAthletics"],
            ["@SparkySunDevil"],
            ["@ASUFootball"],
            ["@ASUAlumni"],
            ["@ASUEngineering"]
        ]
        
        # Common search queries for Facebook content
        search_terms = [
            "event",
            "admission",
            "scholarship",
            "research",
            "innovation",
            "campus life",
            "athletics",
            "graduation",
            "alumni",
            "seminar"
        ]
        
        # 1. Fetch updates from individual accounts
        for account in accounts_to_search:
            result = await self.get_latest_social_media_updates(account_name=account)
            if result and isinstance(result, list):
                results.extend(result)
        
        # 2. Fetch by search terms across all accounts
        for term in search_terms[:5]:  # Limit to first 5 to avoid too many requests
            all_accounts = [item[0] for item in accounts_to_search]
            result = await self.get_latest_social_media_updates(
                account_name=all_accounts,
                search_bar_query=term
            )
            if result and isinstance(result, list):
                results.extend(result)
        
        # 3. Specific combinations of accounts and search terms
        combinations = [
            (["@ArizonaState"], "admission"),
            (["@SunDevilAthletics"], "game"),
            (["@ASUFootball"], "season"),
            (["@ASUAlumni"], "networking"),
            (["@ASUEngineering"], "research")
        ]
        
        for accounts, term in combinations:
            result = await self.get_latest_social_media_updates(
                account_name=accounts,
                search_bar_query=term
            )
            if result and isinstance(result, list):
                results.extend(result)
                
        return self.social_media_docs