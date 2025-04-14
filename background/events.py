from utils.common_imports import *

class Events:
    def __init__(self):
        self.app_config = AppConfig()  
        self.logger = logging.getLogger(__name__)
        self.asu_scraper = ASUWebScraper({}, Utils(vector_store_class=None, asu_data_processor=None, asu_scraper=None, logger=self.logger, group_chat=[]), self.logger)
        self.temp_docs = []
        self.event_docs = []
        self.event_campus_ids = {
            "ASU Downtown": "257211",
            "ASU Online": "257214",
            "ASU Polytechnic": "257212",
            "ASU Tempe": "254417",
            "ASU West Valley": "257213",
            "Fraternity & Sorority Life": "257216",
            "Housing & Residential Life": "257215"
        }
        
        self.event_category_ids = {
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
        
        self.event_theme_ids = {
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
        
        self.event_perk_ids = {
            "Credit": "Credit",
            "Free Food": "FreeFood",
            "Free Stuff": "FreeStuff"
        }
    
    async def get_latest_event_updates(self, search_bar_query: str = None, event_category: list = None, 
                                event_theme: list = None, event_campus: list = None, 
                                shortcut_date: str = None, event_perk: list = None):
        
        if not any([search_bar_query, event_category, event_theme, event_campus]):
            return "At least one parameter of this function is required. Neither Search query and organization category and organization campus received. Please provide at least one parameter to perform search."
        
        search_url = "https://asu.campuslabs.com/engage/events"
        params = []
        
        if event_campus:
            campus_id_array = [self.event_campus_ids[campus] for campus in event_campus if campus in self.event_campus_ids]
            if campus_id_array:
                params.extend([f"branches={campus_id}" for campus_id in campus_id_array])
        
        if event_category:
            category_id_array = [self.event_category_ids[category] for category in event_category if category in self.event_category_ids]
            if category_id_array:
                params.extend([f"categories={category_id}" for category_id in category_id_array])
        
        if event_theme:
            theme_id_array = [self.event_theme_ids[theme] for theme in event_theme if theme in self.event_theme_ids]
            if theme_id_array:
                params.extend([f"themes={theme_id}" for theme_id in theme_id_array])
        
        if event_perk:
            perk_id_array = [self.event_perk_ids[perk] for perk in event_perk if perk in self.event_perk_ids]
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
        
        try:
            self.temp_docs = await self.asu_scraper.engine_search(search_url, doc_title)
            self.event_docs.append({
                "documents": self.temp_docs,
                "search_context": f"ASU Event information for {doc_title}",
                "title": doc_title
            })
            
            return self.event_docs
        except Exception as e:
            return f"Error performing event search: {str(e)}"
    
    async def perform_web_search(self):
        """
        Comprehensive background scraping that tries multiple combinations of search terms,
        categories, themes, campuses, and perks to gather maximum event information.
        """
        results = []
        
        # Common search terms for events
        search_terms = ["workshop", "concert", "seminar", "career fair", 
                      "festival", "lecture", "social", "networking", "hackathon", "exhibition"]
        
        # Get various options
        all_campuses = list(self.event_campus_ids.keys())
        all_categories = list(self.event_category_ids.keys())[:10]  # Limit to 10 for efficiency
        all_themes = list(self.event_theme_ids.keys())
        all_perks = list(self.event_perk_ids.keys())
        all_dates = ["tomorrow", "this_weekend"]
        
        # 1. Fetch by individual search terms
        for term in search_terms:
            results.append(await self.get_latest_event_updates(search_bar_query=term))
        
        # 2. Fetch by individual categories
        for category in all_categories:
            results.append(await self.get_latest_event_updates(event_category=[category]))
        
        # 3. Fetch by individual themes
        for theme in all_themes:
            results.append(await self.get_latest_event_updates(event_theme=[theme]))
        
        # 4. Fetch by individual campuses
        for campus in all_campuses:
            results.append(await self.get_latest_event_updates(event_campus=[campus]))
        
        # 5. Fetch by dates
        for date in all_dates:
            results.append(await self.get_latest_event_updates(shortcut_date=date))
        
        # 6. Fetch by perks
        for perk in all_perks:
            results.append(await self.get_latest_event_updates(event_perk=[perk]))
        
        # 7. Selected combinations
        # Search term + category
        for term, category in zip(search_terms[:3], all_categories[:3]):
            results.append(await self.get_latest_event_updates(
                search_bar_query=term,
                event_category=[category]
            ))
        
        # Search term + theme
        for term, theme in zip(search_terms[3:6], all_themes[:3]):
            results.append(await self.get_latest_event_updates(
                search_bar_query=term,
                event_theme=[theme]
            ))
        
        # Campus + date + perk
        for campus, date, perk in zip(all_campuses[:3], all_dates, all_perks):
            results.append(await self.get_latest_event_updates(
                event_campus=[campus],
                shortcut_date=date,
                event_perk=[perk]
            ))
        
        # 8. More complex combinations
        combinations = [
            ("career fair", ["Career and Professional Development"], ["ThoughtfulLearning"], ["ASU Tempe"], "tomorrow", ["Free Food"]),
            ("concert", ["Student Organization Event"], ["Arts"], ["ASU Downtown"], "this_weekend", ["Free Stuff"]),
            ("workshop", ["Entrepreneurship & Innovation"], ["Social"], ["ASU Polytechnic"], "tomorrow", ["Credit"])
        ]
        
        for term, category, theme, campus, date, perk in combinations:
            results.append(await self.get_latest_event_updates(
                search_bar_query=term,
                event_category=category,
                event_theme=theme,
                event_campus=campus,
                shortcut_date=date,
                event_perk=perk
            ))
        
        return self.event_docs