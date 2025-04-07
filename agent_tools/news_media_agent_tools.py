from utils.common_imports import *

class News_Media_Agent_Tools:
    
    def __init__(self,firestore,utils,logger):
        self.firestore = firestore
        self.utils = utils
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3
        
    async def get_latest_news_updates(self, news_campus : list = None, search_bar_query: str = None,):
        if not any([search_bar_query, news_campus]):
            return "At least one parameter of this function is required. Neither Search query and news campus received. Please provide at least one parameter to perform search."
        
        search_url = "https://asu.campuslabs.com/engage/news"
        params = []
        news_campus_ids = { "ASU Downtown":"257211","ASU Online":"257214","ASU Polytechnic":"257212","ASU Tempe":"254417","ASU West Valley":"257213","Fraternity & Sorority Life":"257216","Housing & Residential Life":"257215"}
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif news_campus:
            doc_title = " ".join(news_campus)
        else:
            doc_title = None

        if news_campus:
            campus_id_array = [news_campus_ids[campus] for campus in news_campus if campus in news_campus_ids]
            if campus_id_array:
                params.extend([f"branches={campus_id}" for campus_id in campus_id_array])
                
        if search_bar_query:
            params.append(f"query={search_bar_query.lower().replace(' ', '%20')}")
        
        if params:
            search_url += "?" + "&".join(params)
        
        return await self.utils.perform_web_search(search_url,doc_title=doc_title, doc_category="news_info")
    
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
                "https://www.instagram.com/arizonastateuniversity"
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
    