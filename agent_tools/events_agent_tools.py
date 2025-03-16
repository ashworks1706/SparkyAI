from utils.common_imports import *

class Events_Agent_Tools:
    def __init__(self,firestore,utils,logger):
        self.firestore = firestore
        self.utils = utils
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3
        

    async def get_latest_event_updates(self, search_bar_query: str = None, event_category: list = None, 
                               event_theme: list = None, event_campus: list = None, 
                               shortcut_date: str = None, event_perk: list = None):
        
        if not any([search_bar_query, event_category, event_theme, event_campus]):
            return "At least one parameter of this function is required. Neither Search query and organization category and organization campus received. Please provide at least one parameter to perform search."
        
        search_url = "https://asu.campuslabs.com/engage/events"
        params = []
        
        event_campus_ids = {
            "ASU Downtown": "257211",
            "ASU Online": "257214",
            "ASU Polytechnic": "257212",
            "ASU Tempe": "254417",
            "ASU West Valley": "257213",
            "Fraternity & Sorority Life": "257216",
            "Housing & Residential Life": "257215"
        }
        
        event_category_ids = {
            "ASU New Student Experience": "18002",
            "ASU Sync": "15695",
            "ASU Welcome Event": "12897",
            "Barrett Student Organization": "12902",
            "Black History Month": "21730",
            "C3": "19049",
            "Career and Professional Development": "12885",
            "Change The World": "12887",
            "Changemaker Central": "12886",
            "Civic Engagement": "17075",
            "Club Meetings": "12887",
            "Clubs and Organization Information": "12888",
            "Community Service": "12903",
            "Cultural Connections and Multicultural community of Excellence": "21719",
            "Culture @ ASU": "12898",
            "DeStress Fest": "19518",
            "Entrepreneurship & Innovation": "17119",
            "General": "12889",
            "Graduate": "12906",
            "Hispanic Heritage Month": "21723",
            "Homecoming": "20525",
            "In-Person Event": "17447",
            "International": "12899",
            "Memorial Union & Student Pavilion Programs": "12900",
            "Multicultural community of Excellence": "19389",
            "PAB Event": "12890",
            "Salute to Service": "12891",
            "Student Engagement Event": "12892",
            "Student Organization Event": "12893",
            "Sun Devil Athletics": "12894",
            "Sun Devil Civility": "12901",
            "Sun Devil Fitness/Wellness": "12895",
            "Sustainability": "12905",
            "University Signature Event": "12904",
            "W.P. Carey Event": "17553"
        }
        
        event_theme_ids = {
            "Arts": "arts",
            "Athletics": "athletics",
            "Community Service": "community_service",
            "Cultural": "cultural",
            "Fundraising": "fundraising",
            "GroupBusiness": "group_business",
            "Social": "social",
            "Spirituality": "spirituality",
            "ThoughtfulLearning": "thoughtful_learning"
        }
        
        event_perk_ids = {
            "Credit": "Credit",
            "Free Food": "FreeFood",
            "Free Stuff": "FreeStuff"
        }
        
        if event_campus:
            campus_id_array = [event_campus_ids[campus] for campus in event_campus if campus in event_campus_ids]
            if campus_id_array:
                params.extend([f"branches={campus_id}" for campus_id in campus_id_array])
        
        if event_category:
            category_id_array = [event_category_ids[category] for category in event_category if category in event_category_ids]
            if category_id_array:
                params.extend([f"categories={category_id}" for category_id in category_id_array])
        
        if event_theme:
            theme_id_array = [event_theme_ids[theme] for theme in event_theme if theme in event_theme_ids]
            if theme_id_array:
                params.extend([f"themes={theme_id}" for theme_id in theme_id_array])
        
        if event_perk:
            perk_id_array = [event_perk_ids[perk] for perk in event_perk if perk in event_perk_ids]
            if perk_id_array:
                params.extend([f"perks={perk_id}" for perk_id in perk_id_array])
        
        if shortcut_date:
            valid_dates = ["tomorrow", "this_weekend"]
            if shortcut_date.lower() in valid_dates:
                params.append(f"shortcutdate={shortcut_date.lower()}")
        
        if search_bar_query:
            params.append(f"query={search_bar_query.lower().replace(' ', '%20')}")
        
        if params:
            search_url += "?" + "&".join(params)
        
        doc_title=""
        if search_bar_query:
            doc_title = search_bar_query
        elif event_category:
            doc_title = " ".join(event_category)
        elif event_theme:
            doc_title = " ".join(event_theme)
        elif event_campus:
            doc_title = " ".join(event_campus)
        elif shortcut_date:
            doc_title = shortcut_date
        elif event_perk:
            doc_title = " ".join(event_perk)
        else:
            doc_title = None
        
        return await self.utils.perform_web_search(search_url, doc_title=doc_title, doc_category = "events_info")
  