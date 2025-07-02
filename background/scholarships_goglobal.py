from utils.common_imports import *

class Scholarships_GoGlobal:
    def __init__(self):
        self.app_config = AppConfig()
        self.logger = logging.getLogger(__name__)
        self.scraper = ASUWebScraper({}, Utils(vector_store_class=None, asu_data_processor=None, asu_scraper=None, logger=self.logger, group_chat=[]), self.logger)
        self.temp_docs = []
        self.scholarship_docs = []

    async def get_latest_scholarships(self, search_bar_query: str = None, academic_level:str = None,
                                      eligible_applicants: str = None, citizenship_status: str = None,
                                      gpa: str = None, focus: str = None):
        
        if not any([search_bar_query, academic_level, citizenship_status, gpa, eligible_applicants, focus]):
            return "Please provide at least one parameter to perform the search."
        
        results = []
        doc_title = ""
        if search_bar_query:
            doc_title = search_bar_query
        elif academic_level:
            doc_title = academic_level
        elif citizenship_status:
            doc_title = citizenship_status
        elif focus:
            doc_title = focus
        else:
            doc_title = None
            
        # First source: GoGlobal ASU
        search_url = f"https://goglobal.asu.edu/scholarship-search"
        query = f"academiclevel={academic_level}&citizenship_status={citizenship_status}&gpa={gpa}"
        
        result = await self.scraper.engine_search(search_url, query)
        results.append(result)
        
        self.scholarship_docs.append({
                    "documents": results,
                    "search_context": f"Scholarships updates for {doc_title}",
                    "title": doc_title,
                })
        
        
        return results
    
    async def perform_web_search(self):
        """
        Comprehensive background scraping for scholarships from multiple sources
        to gather maximum scholarship information.
        """
        results = []
        
        # Common academic levels
        academic_levels = ["undergraduate", "graduate", "doctoral", "postdoctoral"]
        
        # Common citizenship statuses
        citizenship_statuses = ["us_citizen", "permanent_resident", "international"]
        
        # Common GPAs
        gpas = ["2.5", "3.0", "3.5", "4.0"]
        
        # Common focus areas
        focus_areas = [
            "engineering", "business", "science", "arts", "humanities",
            "medicine", "law", "education", "sustainability", "technology"
        ]
        
        # Common eligible applicants
        eligible_applicants = [
            "first_generation", "minority", "women", "veteran", "transfer",
            "returning", "lgbtq", "disability", "international"
        ]
        
        # Common search queries
        search_terms = [
            "full_tuition", "merit", "need_based", "study_abroad",
            "research", "internship", "fellowship"
        ]
        
        # 1. Search by academic levels
        for level in academic_levels:
            result = await self.get_latest_scholarships(academic_level=level)
            
        
        # 2. Search by citizenship status
        for status in citizenship_statuses:
            result = await self.get_latest_scholarships(citizenship_status=status)
            
        
        # 3. Search by focus areas (limited to first 5)
        for focus in focus_areas[:5]:
            result = await self.get_latest_scholarships(focus=focus)
            
        
        # 4. Search by specific combinations
        combinations = [
            (None, "undergraduate", None, "us_citizen", "3.5", None),  # High GPA US undergrads
            (None, "graduate", None, "international", None, "engineering"),  # International engineering grad students
            ("merit", None, "minority", None, None, None),  # Merit scholarships for minorities
            ("research", "doctoral", None, None, None, "science"),  # Research doctorates in science
            (None, None, "women", None, None, "technology")  # Women in technology
        ]
        
        for search, level, eligible, citizenship, gpa, focus in combinations:
            result = await self.get_latest_scholarships(
                search_bar_query=search,
                academic_level=level,
                eligible_applicants=eligible,
                citizenship_status=citizenship,
                gpa=gpa,
                focus=focus
            )
            
        
        # 5. General search terms
        for term in search_terms[:3]:  # Limit to first 3 to avoid too many requests
            result = await self.get_latest_scholarships(search_bar_query=term)
            
                
        return self.scholarship_docs