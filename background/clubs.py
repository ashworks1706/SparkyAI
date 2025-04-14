from utils.common_imports import *


class Clubs:
    def __init__(self):
        self.app_config = AppConfig()  
        self.logger = logging.getLogger(__name__)
        self.asu_scraper = ASUWebScraper({}, Utils(vector_store_class=None, asu_data_processor=None, asu_scraper=None, logger=self.logger,group_chat=[]), self.logger)
        self.temp_docs = []
        self.club_docs = []
        
    
    async def get_latest_club_information(self, search_bar_query: str = None, organization_category: list = None, organization_campus: list = None):
        if not any([search_bar_query, organization_category, organization_campus]):
            return "At least one parameter of this function is required. Neither Search query and organization category and organization campus received. Please provide at least one parameter to perform search."
        
        search_url = "https://asu.campuslabs.com/engage/organizations"
        params = []
        organization_campus_ids = {
            "ASU Downtown": "257211",
            "ASU Online": "257214",
            "ASU Polytechnic": "257212",
            "ASU Tempe": "254417",
            "ASU West Valley": "257213",
            "Fraternity & Sorority Life": "257216",
            "Housing & Residential Life": "257215"
        }
        
        organization_category_ids = {
            "Academic": "13382", "Barrett": "14598", "Creative/Performing Arts": "13383",
            "Cultural/Ethnic": "13384", "Distinguished Student Organization": "14549",
            "Fulton Organizations": "14815", "Graduate": "13387", "Health/Wellness": "13388",
            "International": "13389", "LGBTQIA+": "13391", "Political": "13392",
            "Professional": "13393", "Religious/Faith/Spiritual": "13395", "Service": "13396",
            "Social Awareness": "13398", "Special Interest": "13399", "Sports/Recreation": "13400",
            "Sustainability": "13402", "Technology": "13403", "Veteran Groups": "14569",
            "W.P. Carey Organizations": "14814", "Women": "13405"
        }
        
        doc_title = ""
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
        
        try:
            self.temp_docs = await self.asu_scraper.engine_search(search_url, doc_title)
            self.club_docs.append({
                "documents": self.temp_docs,
                "search_context": f"ASU Club information for {search_bar_query}",
                "title": doc_title
            })
            
            return self.club_docs
        except Exception as e:
            return f"Error performing club search: {str(e)}"
    
    async def perform_web_search(self):
        """
        Comprehensive background scraping that tries multiple combinations of search terms,
        categories, and campuses to gather maximum club information.
        """
        results = []
        
        # Common search terms for clubs
        search_terms = ["academic", "engineering", "robotics", "arts", "business", 
                    "science", "volunteer", "sustainability", "sports", "cultural"]
        
        # Get all campus and category options
        all_campuses = list(self.organization_campus_ids.keys())
        all_categories = list(self.organization_category_ids.keys())
        
        # 1. Fetch by individual search terms
        for term in search_terms:
            results.append(await self.get_latest_club_information(search_bar_query=term))
        
        # 2. Fetch by individual categories
        for category in all_categories:
            results.append(await self.get_latest_club_information(organization_category=[category]))
        
        # 3. Fetch by individual campuses
        for campus in all_campuses:
            results.append(await self.get_latest_club_information(organization_campus=[campus]))
        
        # 4. Selected combinations of categories
        category_pairs = [
            ["Technology", "Academic"], 
            ["Creative/Performing Arts", "Cultural/Ethnic"],
            ["Professional", "Graduate"],
            ["Sports/Recreation", "Health/Wellness"],
            ["Service", "Sustainability"]
        ]
        for pair in category_pairs:
            results.append(await self.get_latest_club_information(organization_category=pair))
        
        # 5. Selected combinations of search terms + categories
        for term, category in zip(search_terms[:5], all_categories[:5]):
            results.append(await self.get_latest_club_information(
                search_bar_query=term,
                organization_category=[category]
            ))
        
        # 6. Selected combinations of search terms + campuses
        for term, campus in zip(search_terms[5:], all_campuses[:5]):
            results.append(await self.get_latest_club_information(
                search_bar_query=term,
                organization_campus=[campus]
            ))
        
        # 7. Selected full combinations (search term + category + campus)
        combinations = [
            ("engineering", ["Technology"], ["ASU Polytechnic"]),
            ("arts", ["Creative/Performing Arts"], ["ASU Tempe"]),
            ("business", ["Professional"], ["ASU Downtown"]),
            ("online", ["Special Interest"], ["ASU Online"])
        ]
        for term, category, campus in combinations:
            results.append(await self.get_latest_club_information(
                search_bar_query=term,
                organization_category=category,
                organization_campus=campus
            ))
        
        
        return self.club_docs
