from utils.common_imports import *
from datetime import datetime
import re
from urllib.parse import quote_plus


class Campus_Agent_Tools:
    """
    Utilities for querying the ASU interactive map from other agents
    and returning a clean “<Building Name> — Google‑maps: <URL>” string.
    """

    def __init__(self, middleware, utils, logger):
        self.middleware = middleware
        self.utils = utils
        self.logger = logger
        self.text_content = []

    # --------------------------------------------------------------------- #
    # MAIN PUBLIC METHOD
    # --------------------------------------------------------------------- #
    async def get_campus_location(self, keyword: str) -> str:
        """
        Look up a building / landmark on the ASU interactive map.

        

        Returns
        -------
        str
            • “<Building Name> — Google‑maps: <url>”  – if full name + link found  
            • “Google‑maps: <url>”                    – if only link parsed  
            • raw map‑scrape text                     – if parsing failed
        """
        self.logger.info(f"Campus_Agent_Tools: looking up '{keyword}'")

        # 1) run Selenium scraper (wrapped by utils.perform_web_search)
        result_text = await self.utils.perform_web_search(
            "https://www.asu.edu/map/interactive/",
            optional_query={"loc": keyword, "details": "summary"},
            doc_title=f"Campus Map Lookup: {keyword}",
            doc_category="campus_info"
        )
        if not isinstance(result_text, str):
            # perform_web_search sometimes returns list[dict]
            result_text = result_text[0]["content"] if result_text else ""

        # 2) normalise text
        cleaned = re.sub(r"[ \t]+", " ", result_text.strip())  # collapse NBSP & tabs
        lines   = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]

        building_name, gmaps_url = None, None
        lat, lng = None, None

        # 3) parse each line
        for ln in lines:
            low = ln.lower()

            # “Building : …”
            if low.startswith("building"):
                building_name = ln.split(":", 1)[-1].strip()

            # “CODE – Full Name”  (e.g. “BYAC – Business Administration C Wing”)
            m = re.match(r"^([A-Z]{3,4})\s+[-–]\s+(.+)", ln)
            if m:
                building_name = m.group(2).strip()          # full descriptive name
                # if you want to include the code too:
                # building_name = f"{m.group(1)} – {m.group(2).strip()}"

            # explicit Google‑maps link
            if "maps.google.com" in ln:
                gmaps_url = ln.split(":", 1)[-1].strip()

            # bare coordinates “33.4195, -111.93”
            coords = re.findall(r"(-?\d{2}\.\d+)", ln)
            if len(coords) >= 2:
                lat, lng = coords[0], coords[1]

        # 4) construct link if scraper only gave lat/lng
        if not gmaps_url and lat and lng:
            gmaps_url = f"https://maps.google.com/?q={lat},{lng}"

        # 5) final nicely formatted string
        if building_name and gmaps_url:
            return f"{building_name} — Google‑maps: {gmaps_url}"
        
        else:
            self.logger.warning(
                "Campus_Agent_Tools: could not parse map response; returning raw text."
            )
            return result_text
