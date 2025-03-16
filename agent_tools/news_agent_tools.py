from utils.common_imports import *

class News_Agent_Tools:
    def __init__(self,firestore,utils,logger):
        self.firestore = firestore
        self.utils = utils
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3
        
  
    async def get_latest_news_updates(self, news_campus : list = None, search_bar_query: str = None,):
        if not any([search_bar_query, news_campus]):
            return "At least one parameter of this function is required. Neither Search query and news campus received. Please provide at least one parameter to perform search."
        
        search_url = "https://asu.campuslabs.com/engage/organizations"
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
   