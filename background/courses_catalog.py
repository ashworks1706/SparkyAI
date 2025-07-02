from utils.common_imports import *

class Courses_Catalog:
    def __init__(self):
        self.app_config = AppConfig()  
        self.logger = logging.getLogger(__name__)
        self.asu_scraper = ASUWebScraper({}, Utils(vector_store_class=None, asu_data_processor=None, asu_scraper=None, logger=self.logger, group_chat=[]), self.logger)
        self.temp_docs = []
        self.courses_docs = []
        self.term_ids = {
            "Fall 2023": "2237",
            "Spring 2024": "2241",
            "Summer 2024": "2244",
            "Fall 2024": "2247",
            "Spring 2025": "2251"
        }

    async def get_latest_class_information(self, search_bar_query: Optional[str] = None, class_term: Optional[str] = None, subject: Optional[str] = None):
        if not any([search_bar_query, class_term, subject]):
            return "At least one parameter of this function is required. Please provide a search query, class term, or subject to perform search."
        
        search_url = "https://catalog.apps.asu.edu/catalog/classes/search"
        params = []
        doc_title = ""
        
        if search_bar_query:
            doc_title = search_bar_query
        elif subject:
            doc_title = subject
        elif class_term:
            doc_title = class_term
        else:
            doc_title = None

        if class_term and class_term in self.term_ids:
            params.append(f"term={self.term_ids[class_term]}")
                
        if search_bar_query:
            params.append(f"keywords={search_bar_query.lower().replace(' ', '%20')}")
            
        if subject:
            params.append(f"subject={subject.upper()}")
        
        if params:
            search_url += "?" + "&".join(params)
        
        try:
            self.temp_docs = await self.asu_scraper.engine_search(search_url, doc_title)
            self.courses_docs.append({
                "documents": self.temp_docs,
                "search_context": f"ASU Course information for {doc_title}",
                "title": doc_title
            })
            
            return self.courses_docs
        except Exception as e:
            return f"Error performing course search: {str(e)}"
    
    async def perform_web_search(self):
        """
        Comprehensive background scraping for courses that tries multiple combinations 
        of search terms, subjects, and terms to gather maximum course information.
        """
        results = []
        
        # Common subjects
        common_subjects = ["CSE", "ENG", "MAT", "PHY", "CHM", "BIO", "PSY", "SOC", "ECN", "HST"]
        
        # Common search terms for courses
        search_terms = ["artificial intelligence", "data science", "machine learning", 
                       "programming", "algorithms", "research methods", "statistics", 
                       "sustainability", "innovation", "design"]
        
        # Get all terms
        all_terms = list(self.term_ids.keys())
        
        # 1. Fetch by individual subjects
        for subject in common_subjects[:5]:  # Limit to first 5 to avoid too many requests
            results.append(await self.get_latest_class_information(subject=subject))
        
        # 2. Fetch by individual terms
        for term in all_terms[:3]:  # Latest 3 terms
            results.append(await self.get_latest_class_information(class_term=term))
        
        # 3. Fetch by individual search terms
        for term in search_terms[:5]:  # Limit to first 5 terms
            results.append(await self.get_latest_class_information(search_bar_query=term))
        
        # 4. Selected combinations of subjects and terms
        combinations = [
            ("CSE", "Fall 2024"),
            ("MAT", "Spring 2024"),
            ("ENG", "Summer 2024"),
            ("PHY", "Fall 2024"),
            ("BIO", "Spring 2024")
        ]
        
        for subject, term in combinations:
            results.append(await self.get_latest_class_information(
                subject=subject,
                class_term=term
            ))
            
        # 5. Search terms with specific terms
        complex_combinations = [
            ("artificial intelligence", "Fall 2024"),
            ("data science", "Spring 2024"),
            ("programming", "Summer 2024")
        ]
        
        for search, term in complex_combinations:
            results.append(await self.get_latest_class_information(
                search_bar_query=search,
                class_term=term
            ))
            
        # 6. Full combinations (search term, subject, term)
        full_combinations = [
            ("algorithms", "CSE", "Fall 2024"),
            ("research methods", "PSY", "Spring 2024"),
            ("sustainability", "SOC", "Fall 2024")
        ]
        
        for search, subject, term in full_combinations:
            results.append(await self.get_latest_class_information(
                search_bar_query=search,
                subject=subject,
                class_term=term
            ))
            
        return self.courses_docs