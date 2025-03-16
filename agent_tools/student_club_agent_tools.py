from utils.common_imports import *

class Student_Club_Agent_Tools:
    def __init__(self,firestore,utils,logger):
        self.firestore = firestore
        self.utils = utils
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3
        
    async def get_latest_club_information(self, search_bar_query: str = None, organization_category: list = None, organization_campus: list = None):
        if not any([search_bar_query, organization_category, organization_campus]):
            return "At least one parameter of this function is required. Neither Search query and organization category and organization campus received. Please provide at least one parameter to perform search."
        
        search_url = "https://asu.campuslabs.com/engage/organizations"
        params = []
        organization_campus_ids = { "ASU Downtown":"257211",
                                   "ASU Online":"257214",
                                   "ASU Polytechnic":"257212",
                                   "ASU Tempe":"254417",
                                   "ASU West Valley":"257213",
                                   "Fraternity & Sorority Life":"257216",
                                   "Housing & Residential Life":"257215"}
        
        organization_category_ids = {"Academic":"13382","Barrett":"14598","Creative/Performing Arts":"13383","Cultural/Ethnic":"13384","Distinguished Student Organization":"14549","Fulton Organizations":"14815","Graduate":"13387","Health/Wellness":"13388","International":"13389","LGBTQIA+":"13391","Political":"13392","Professional":"13393","Religious/Faith/Spiritual":"13395","Service":"13396","Social Awareness":"13398","Special Interest":"13399","Sports/Recreation":"13400","Sustainability":"13402","Technology":"13403", "Veteran Groups":"14569","W.P. Carey Organizations":"14814","Women":"13405"}
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif organization_category:
            doc_title = " ".join(organization_category)
        elif organization_campus:
            doc_title = " ".join(organization_campus)
        else:
            doc_title = None

 
        if organization_campus:
            campus_id_array = [organization_campus_ids[campus] for campus in organization_campus if campus in organization_campus_ids]
            if campus_id_array:
                params.extend([f"branches={campus_id}" for campus_id in campus_id_array])
        
        if organization_category:
            category_id_array = [organization_category_ids[category] for category in organization_category if category in organization_category_ids]
            if category_id_array:
                params.extend([f"categories={category_id}" for category_id in category_id_array])
        
        if search_bar_query:
            params.append(f"query={search_bar_query.lower().replace(' ', '%20')}")
        
        if params:
            search_url += "?" + "&".join(params)
        
        return await self.utils.perform_web_search(search_url, doc_title=doc_title, doc_category ="clubs_info")
        
  
    