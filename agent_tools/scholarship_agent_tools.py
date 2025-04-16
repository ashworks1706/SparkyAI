from utils.common_imports import *

class Scholarship_Agent_Tools:
    def __init__(self,middleware,utils,logger):
        self.middleware = middleware
        self.utils = utils
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3
    
        
    async def get_latest_scholarships(self, search_bar_query: str = None, academic_level:str = None,eligible_applicants: str =None, citizenship_status: str = None, gpa: str = None, focus : str = None):
    
        if not any([search_bar_query, academic_level, citizenship_status, gpa,  eligible_applicants, focus]):
            return "Please provide at least one parameter to perform the search."
        
        results =[]
        doc_title = ""
        if search_bar_query:
            doc_title = search_bar_query
        elif academic_level:
            doc_title = academic_level
        elif citizenship_status:
            doc_title = citizenship_status
        # elif college:
        #     doc_title = college
        elif focus:
            doc_title = focus
        else:
            doc_title = None
            
        
        search_url = f"https://goglobal.asu.edu/scholarship-search"
        
        query = f"academiclevel={academic_level}&citizenship_status={citizenship_status}&gpa={gpa}"
        # &college={college}
        
        result = await self.utils.perform_web_search(search_url,query, doc_title=doc_title, doc_category ="scholarships_info")
        
        
        results.append(result)
        
        
        search_url = f"https://onsa.asu.edu/scholarships"
        
        query = f"search_bar_query={search_bar_query}&citizenship_status={citizenship_status}&eligible_applicants={eligible_applicants}&focus={focus}"
        
        
        
        result = await self.utils.perform_web_search(search_url,query, doc_title=doc_title,doc_category ="scholarships_info")
        
        
        results.append(result)
        
            
        
        return results
   