import os
import asyncio
from typing import List, Dict, Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from langchain_community.document_loaders import UnstructuredURLLoader


class ASUWebScraper:
    """
    Scrapes ASU Engage pages—including the new SunDevilCentral portal—without manual login.
    """

    def __init__(self, middleware, utils, logger):
        self.middleware = middleware
        self.utils = utils
        self.logger = logger

        # Minimal headers for requests
        self.headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml",
        }

        # Headless Selenium for fallback
        opts = Options()
        opts.add_argument("--headless")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1920,1080")

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=opts
        )

    async def engine_search(
        self,
        search_url: str = None,
        optional_query: str = None
    ) -> List[Dict[str, Any]]:
        if not search_url:
            return []

        self.logger.info(f"ASUWebScraper.engine_search → {search_url}")

        if "sundevilcentral.eoss.asu.edu" in search_url:
            return await self._scrape_sundevilcentral(search_url)
        else:
            return await self._scrape_rendered_page(search_url)

    async def _scrape_sundevilcentral(self, url: str) -> List[Dict[str, Any]]:
        """
        Try a plain HTTP scrape of SunDevilCentral. If we detect a login form,
        submit credentials automatically and then retry.
        """
        docs: List[Dict[str, Any]] = []

        def _http_get(u):
            return requests.get(u, headers=self.headers, timeout=15)

        try:
            res = _http_get(url)
            html = res.text

            # Detect login form
            if "name=\"username\"" in html and "name=\"password\"" in html:
                self.logger.info("Detected SunDevilCentral login form; submitting credentials")
                login_payload = {
                    "username": os.getenv("SDC_USER", ""),
                    "password": os.getenv("SDC_PASS", ""),
                }
                # Post to the same URL (the home_login form posts to itself)
                res = requests.post(url, data=login_payload, headers=self.headers, timeout=15)
                html = res.text

            soup = BeautifulSoup(html, "html.parser")
            # Find any club/event links in cards or tables:
            anchors = soup.select("a[href*='/club_signup'], a[href*='/events/']")
            seen = set()
            for a in anchors:
                href = a.get("href")
                if not href:
                    continue
                full = href if href.startswith("http") else urljoin(url, href)
                if full in seen:
                    continue
                seen.add(full)
                # Delegate to our normal renderer for each detail page
                docs.extend(await self._scrape_rendered_page(full))

        except Exception as e:
            self.logger.error(f"Error scraping SunDevilCentral at {url}: {e}")

        return docs

    async def _scrape_rendered_page(self, url: str) -> List[Dict[str, Any]]:
        """
        Render the URL with Selenium, then funnel into UnstructuredURLLoader.
        Falls back to raw page_source if that loader fails or returns no text.
        """
        result: List[Dict[str, Any]] = []

        try:
            # 1) Let Selenium fetch & render JS
            self.driver.get(url)
            await asyncio.sleep(3)

            # 2) Try UnstructuredURLLoader
            try:
                loader = UnstructuredURLLoader(urls=[url])
                docs = loader.load()
                text = docs[0].page_content.strip()
                if not text:
                    raise ValueError("empty content")
            except Exception as e:
                self.logger.warning(f"UnstructuredURLLoader failed on {url}: {e}")
                text = self.driver.page_source

            result.append({
                "content": text,
                "metadata": {"url": url}
            })

        except Exception as e:
            self.logger.error(f"Error rendering {url}: {e}")

        return result
