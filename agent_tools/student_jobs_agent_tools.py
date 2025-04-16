from utils.common_imports import *
from typing import List, Dict
from rag.web_scrape import ASUWebScraper  # Adjust the import path to where your web_scrape.py is located
from datetime import datetime

class Student_Jobs_Agent_Tools:
    def __init__(self, middleware, utils, logger):
        self.middleware = middleware
        self.utils = utils
        self.logger = logger
        self.text_content = []

    async def get_workday_student_jobs(self, keyword: str, max_results: int = 5) -> str:
        """
        This tool function is invoked by the agent when it needs to scrape ASU Workday jobs.
        It instantiates the ASUWebScraper, calls its scrape_asu_workday_jobs method, and then
        formats the returned job data into a single combined text string.
        """
        self.logger.info(f"Scraping ASU Workday for keyword: {keyword} with max results: {max_results}")
        
        optional_query = {
            "keyword": keyword,
            "max_results": max_results
        }
        
        result = await self.utils.perform_web_search("https://www.myworkday.com/asu",optional_query=optional_query, doc_title="ASU Workday Jobs", doc_category="student_jobs")
        
        return result
