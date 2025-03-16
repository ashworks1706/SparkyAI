from utils.common_imports import *

class Social_Media_Agent_Tools:
    def __init__(self,firestore,utils,logger):
        self.firestore = firestore
        self.utils = utils
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3

    async def get_latest_social_media_updates(self,  account_name: list, search_bar_query: str = None,):
        if not any([search_bar_query, account_name]):
            return "At least one parameter of this function is required. Neither Search query and news campus received. Please provide at least one parameter to perform search."
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif account_name:
            doc_title = " ".join(account_name)
        else:
            doc_title = None
        account_OBJECTs = {
            "@ArizonaState": [
                "https://x.com/ASU", 
                "https://www.instagram.com/arizonastate"
            ],
            "@SunDevilAthletics": [
                "https://x.com/TheSunDevils", 
            ],
            "@SparkySunDevil": [
                "https://x.com/SparkySunDevil", 
                "https://www.instagram.com/SparkySunDevil"
            ],
            "@ASUFootball": [
                "https://x.com/ASUFootball", 
                "https://www.instagram.com/sundevilfb/"
            ],
            "@ASUFootball": [
                "https://x.com/ASUFootball", 
                "https://www.instagram.com/sundevilfb/"
            ],
        }
        
        # Collect URLs for specified account names
        final_search_array = []
        for name in account_name:
            if name in account_OBJECTs:
                final_search_array.extend(account_OBJECTs[name])
        
        # If no URLs found for specified accounts, return empty list
        if not final_search_array:
            return []
        
        # Perform web search on each URL asynchronously
        search_results = []
        for url in final_search_array:
            search_result = await self.utils.perform_web_search(url,search_bar_query,doc_title=doc_title, doc_category="social_media_updates")
            search_results.extend(search_result)
        return search_results
    