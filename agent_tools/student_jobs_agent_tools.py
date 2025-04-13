from utils.common_imports import *
from typing import List, Dict
from rag.web_scrape import ASUWebScraper  # Adjust the path if necessary

class Student_Jobs_Agent_Tools:
    def __init__(self, firestore, utils, logger):
        self.firestore = firestore
        self.utils = utils
        self.logger = logger
        self.text_content = []

    async def get_workday_student_jobs(self, keyword: str, max_results: int = 5) -> str:
        """
        This tool is called by the agent to scrape Workday jobs.
        Returns a formatted string with the job listings or an error message.
        """
        try:
            scraper = ASUWebScraper(
                discord_state={},  # Adjust with real data if available
                utils=self.utils, 
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(f"@student_jobs_agent_tools.py: Error creating ASUWebScraper: {e}")
            return "Could not initialize the scraper."

        try:
            results = scraper.scrape_asu_workday_jobs(keyword=keyword, max_results=max_results)
        except Exception as e:
            self.logger.error(f"@student_jobs_agent_tools.py: Error scraping Workday jobs: {e}")
            return "Failed to scrape Workday jobs."

        if not results:
            return "No job results found on Workday."

        self.text_content = []
        for idx, job in enumerate(results, start=1):
            title = job.get("title", "N/A")
            header = job.get("detail_header", "N/A")
            detail_text = job.get("detail_text", "N/A")
            # Use only the first 250 characters as an excerpt
            excerpt = detail_text[:250].replace("\n", " ")
            formatted_string = (
                f"Job #{idx}: {title}\n"
                f"Header: {header}\n"
                f"Details Excerpt: {excerpt}...\n"
                "--------------------"
            )
            self.text_content.append({
                'content': formatted_string,
                'metadata': {'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            })

        return "\n\n".join([entry['content'] for entry in self.text_content])
