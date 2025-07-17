# agent_tools/student_clubs_events_tools.py
# --------------------------------------------------------------------------- #
#  ASU RAG – “Student Clubs & Events” tools (SunDevilCentral edition)         #
# --------------------------------------------------------------------------- #

from typing import List, Optional, Dict, Any
from urllib.parse import quote_plus
from utils.common_imports import *  # brings in asyncio, datetime, etc.


class Student_Clubs_Events_Agent_Tools:
    """Tool wrapper used by the Student-Clubs-Events Gemini agent (SunDevilCentral)."""

    def __init__(self, middleware, utils, logger):
        self.middleware = middleware
        self.utils = utils
        self.logger = logger

    # --------------------------------------------------------------------- #
    #  EVENTS                                                               #
    # --------------------------------------------------------------------- #
    async def get_latest_event_updates(
        self,
        search_bar_query: Optional[str] = None,
        event_category: Optional[List[str]] = None,
        event_theme: Optional[List[str]] = None,
        event_campus: Optional[List[str]] = None,
        shortcut_date: Optional[str] = None,
        event_perk: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build a SunDevilCentral *Events* URL from the given filters (if any)
        and return the parsed page via `utils.perform_web_search`.  If *no*
        filters are provided, fetch the default upcoming-events listing.
        """

        base_url = "https://sundevilcentral.eoss.asu.edu/events"
        params: List[str] = []

        # Static look-up tables for filtering if filters are provided
        event_campus_ids = {
            "ASU Tempe": "254417",
            "ASU Downtown": "257211",
            "ASU Polytechnic": "257212",
            "ASU West Valley": "257213",
            "ASU Online": "257214",
            "Fraternity & Sorority Life": "257216",
            "Housing & Residential Life": "257215",
        }
        event_category_ids = {
            "ASU Sync": "15695",
            "Barrett Student Organization": "12902",
            "Community Service": "12903",
            "Cultural": "12898",
            "Entrepreneurship & Innovation": "17119",
            "Graduate": "12906",
            "International": "12899",
            "Student Organization Event": "12893",
            "Sustainability": "12905",
        }
        event_theme_ids = {
            "Arts": "arts",
            "Athletics": "athletics",
            "Community Service": "community_service",
            "Cultural": "cultural",
            "Fundraising": "fundraising",
            "Social": "social",
            "Spirituality": "spirituality",
            "ThoughtfulLearning": "thoughtful_learning",
        }
        event_perk_ids = {
            "Credit": "Credit",
            "Free Food": "FreeFood",
            "Free Stuff": "FreeStuff",
        }

        def _extend(id_map: Dict[str, str], items: List[str], key: str):
            for i in items or []:
                if i in id_map:
                    params.append(f"{key}={id_map[i]}")

        # Only build filters if any were passed in
        if event_campus or event_category or event_theme or event_perk or shortcut_date or search_bar_query:
            _extend(event_campus_ids, event_campus, "branches")
            _extend(event_category_ids, event_category, "categories")
            _extend(event_theme_ids, event_theme, "themes")
            _extend(event_perk_ids, event_perk, "perks")
            if shortcut_date and shortcut_date.lower() in {"tomorrow", "this_weekend"}:
                params.append(f"shortcutdate={shortcut_date.lower()}")
            if search_bar_query:
                params.append(f"query={quote_plus(search_bar_query)}")

        # Final URL either with filters or the default listing
        url = f"{base_url}?{'&'.join(params)}" if params else base_url

        # Title for RAG/vector-store
        title = (
            search_bar_query
            or " ".join(event_category or [])
            or " ".join(event_theme or [])
            or " ".join(event_campus or [])
            or (shortcut_date or "")
            or "upcoming events"
        )

        return await self.utils.perform_web_search(
            url, doc_title=title, doc_category="events_info"
        )

    # --------------------------------------------------------------------- #
    #  CLUBS                                                                #
    # --------------------------------------------------------------------- #
    async def get_latest_club_information(
        self,
        search_bar_query: Optional[str] = None,
        organization_category: Optional[List[str]] = None,
        organization_campus: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build a SunDevilCentral *Organizations* URL from the given filters (if any)
        and return the parsed page via `utils.perform_web_search`. If no filters
        are provided, fetch the default clubs listing.
        """

        base_url = "https://sundevilcentral.eoss.asu.edu/club_signup?view=all&"
        params: List[str] = []

        organization_campus_ids = {
            "ASU Tempe": "254417",
            "ASU Downtown": "257211",
            "ASU Polytechnic": "257212",
            "ASU West Valley": "257213",
            "ASU Online": "257214",
            "Fraternity & Sorority Life": "257216",
            "Housing & Residential Life": "257215",
        }
        organization_category_ids = {
            "Academic": "13382",
            "Barrett": "14598",
            "Creative/Performing Arts": "13383",
            "Cultural/Ethnic": "13384",
            "Service": "13396",
            "Sustainability": "13402",
            "Technology": "13403",
        }

        # Only add filters if provided
        for c in organization_campus or []:
            if c in organization_campus_ids:
                params.append(f"branches={organization_campus_ids[c]}")
        for c in organization_category or []:
            if c in organization_category_ids:
                params.append(f"categories={organization_category_ids[c]}")
        if search_bar_query:
            params.append(f"query={quote_plus(search_bar_query)}")

        url = f"{base_url}?{'&'.join(params)}" if params else base_url

        title = (
            search_bar_query
            or " ".join(organization_category or [])
            or " ".join(organization_campus or [])
            or "student clubs"
        )

        return await self.utils.perform_web_search(
            url, doc_title=title, doc_category="clubs_info"
        )
