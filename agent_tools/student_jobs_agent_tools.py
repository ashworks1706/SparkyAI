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
        max_results = int(max_results)               # ← force to int
        self.logger.info(f"Scraping ASU Workday for keyword: {keyword} with max results: {max_results}")
        optional_query = {
            "keyword": keyword,
            "max_results": max_results
        }
        result = await self.utils.perform_web_search(
            "https://www.myworkday.com/asu",
            optional_query=optional_query,
            doc_title="ASU Workday Jobs",
            doc_category="student_jobs"
        )
        return result
