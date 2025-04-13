from utils.common_imports import *
from typing import List, Dict
from rag.web_scrape import ASUWebScraper  # Adjust the import path to where your web_scrape.py is located
from datetime import datetime

class Student_Jobs_Agent_Tools:
    def __init__(self, firestore, utils, logger):
        self.firestore = firestore
        self.utils = utils
        self.logger = logger
        self.text_content = []

    async def get_workday_student_jobs(self, keyword: str, max_results: int = 5) -> str:
        """
        This tool function is invoked by the agent when it needs to scrape ASU Workday jobs.
        It instantiates the ASUWebScraper, calls its scrape_asu_workday_jobs method, and then
        formats the returned job data into a single combined text string.
        """
        # Instantiate the web scraper
        try:
            scraper = ASUWebScraper(
                discord_state={},  # Pass real data if available
                utils=self.utils,
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(f"@student_jobs_agent_tools.py: Error creating ASUWebScraper: {e}")
            return "Could not initialize the scraper."

        # Call the scraping method; this returns a list of job dictionaries
        try:
            all_jobs = scraper.scrape_asu_workday_jobs(keyword=keyword, max_results=max_results)
        except Exception as e:
            self.logger.error(f"@student_jobs_agent_tools.py: Error scraping Workday jobs: {e}")
            return "Failed to scrape Workday jobs."

        # Check if we got any results
        if not all_jobs:
            return "No job results found on Workday."

        # Format the scraped job data
        self.text_content = []  # reset the text_content array
        for idx, job in enumerate(all_jobs, start=1):
            # Create an excerpt from the detail text (limit to 250 characters)
            snippet = job.get("detail_text", "N/A")[:250].replace("\n", " ")
            # Build a formatted string for the job
            formatted_string = (
                f"Job #{idx}: {job.get('title', 'N/A')}\n"
                f"Header: {job.get('detail_header', 'N/A')}\n"
                f"Details Excerpt: {snippet}...\n"
                "--------------------"
            )
            self.text_content.append({
                'content': formatted_string,
                'metadata': {'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            })

        # Return the combined formatted string (jobs separated by two newlines)
        return "\n\n".join([entry['content'] for entry in self.text_content])
