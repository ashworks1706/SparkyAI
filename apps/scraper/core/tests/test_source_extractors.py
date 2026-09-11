from pathlib import Path

import pytest
from scraper.chunk import chunk_text
from scraper.core.types import Fetched
from scraper.sources import SOURCES
from scraper.sources.clubs import extract_clubs
from scraper.sources.courses import extract_courses
from scraper.sources.events import extract_events
from scraper.sources.jobs import extract_jobs
from scraper.sources.library_hours import extract_hours
from scraper.sources.news import extract_news
from scraper.sources.scholarships import extract_scholarships
from scraper.sources.shuttles import extract_shuttles
from scraper.sources.sports import extract_sports

FIXTURES = Path(__file__).parent / "fixtures"


def page(name: str) -> Fetched:
    markdown = (FIXTURES / name).read_text(encoding="utf-8")
    return Fetched(
        url=f"https://example.invalid/{name}",
        status=200,
        body=markdown.encode("utf-8"),
        content_type="text/markdown",
        text=markdown,
    )


def line_with(text: str, needle: str) -> str:
    matches = [line for line in text.splitlines() if needle in line]
    assert matches, f"no line carries {needle!r}"
    return matches[0]


def test_library_hours_keeps_a_whole_week_on_one_line() -> None:
    text = extract_hours(page("library_hours.md"))
    hayden = line_with(text, "Hayden Library")
    assert "Monday 7 a.m. - 2 a.m." in hayden
    assert "Friday 7 a.m. - 8 p.m." in hayden
    assert "Sunday 10 a.m. - 2 a.m." in hayden
    assert "Saturday Closed" in line_with(text, "Noble Library")
    assert "Monday 7:30 a.m. - 10 p.m." in line_with(text, "Fletcher Library")
    assert "September 8 - September 14, 2025" in text
    assert "|" not in text
    assert "---" not in text
    assert "https://" not in text


def test_events_keep_name_date_and_location_together() -> None:
    text = extract_events(page("events.md"))
    welcome = line_with(text, "Sun Devil Welcome")
    assert "Thursday, August 21, 2025 5:00 p.m. - 7:00 p.m." in welcome
    assert "Old Main Lawn, Tempe campus" in welcome
    fair = line_with(text, "Career Fair")
    assert "September 10, 2025 10:00 a.m. - 3:00 p.m." in fair
    assert "Memorial Union, Ventana Ballroom, Tempe campus" in fair
    showcase = line_with(text, "Graduate Research Showcase")
    assert "Friday, October 3, 2025 1:00 p.m." in showcase
    assert "Online via Zoom" in showcase
    assert "Meet student organizations" in text
    assert "https://" not in text


def test_courses_keep_number_title_and_units_together() -> None:
    text = extract_courses(page("courses.md"))
    assert "CSE 110 | Principles of Programming | 3 units" in text
    assert "CSE 310 | Data Structures and Algorithms | 3 units" in text
    assert "ACC 231 | Uses of Accounting Information I | 3 units" in text
    assert "ACC 494 | Special Topics in Accounting | 1 unit" in text
    assert "Fall 2025" in text
    assert "https://" not in text


def test_clubs_keep_name_category_and_description_together() -> None:
    text = extract_clubs(page("clubs.md"))
    sda = line_with(text, "Software Developers Association")
    assert "Academic and Professional" in sda
    assert "Weekly workshops, hackathon teams" in sda
    robotics = line_with(text, "Sun Devil Robotics Club")
    assert "Engineering and Technology" in robotics
    assert "autonomous robots" in robotics
    kitchen = line_with(text, "Devils in the Kitchen")
    assert "Social and Recreational" in kitchen
    assert "Cooking nights on the Tempe campus" in kitchen
    assert "https://" not in text
    assert "](" not in text


def test_scholarships_keep_award_deadline_and_eligibility_together() -> None:
    text = extract_scholarships(page("scholarships.md"))
    nau = line_with(text, "New American University Scholarship")
    assert "Award $8,500 per year" in nau
    assert "Deadline February 1, 2026" in nau
    assert "Eligibility First-year Arizona residents" in nau
    obama = line_with(text, "Obama Scholars Program")
    assert "Award $12,000 per year" in obama
    assert "Deadline December 1, 2025" in obama
    assert "Eligibility Open to Arizona residents" in obama
    fulton = line_with(text, "Fulton Undergraduate Research Award")
    assert "Award $1,500" in fulton
    assert "Deadline October 15, 2025" in fulton
    assert "Eligibility Engineering majors with a 3.0 GPA" in fulton
    barrett = line_with(text, "Barrett Study Abroad Grant")
    assert "Award $2,000" in barrett
    assert "Deadline November 7, 2025" in barrett
    assert "**" not in text
    assert "https://" not in text


def test_news_keeps_headline_date_and_summary_together() -> None:
    text = extract_news(page("news.md"))
    biodesign = line_with(text, "biodesign building")
    assert "September 3, 2025" in biodesign
    assert "188,000-square-foot lab" in biodesign
    solar = line_with(text, "solar car race")
    assert "August 28, 2025" in solar
    assert "thirty undergraduates" in solar
    enrollment = line_with(text, "Fall enrollment passes 150,000 students")
    assert "August 21, 2025" in enrollment
    assert "Online enrollment grew by 6 percent" in enrollment
    assert "https://" not in text
    assert "](" not in text


def test_shuttles_keep_route_stops_and_running_times_together() -> None:
    text = extract_shuttles(page("shuttles.md"))
    downtown = line_with(text, "Tempe - Downtown Phoenix")
    assert "Stops Tempe Transit Center, Taylor Place, University Center" in downtown
    assert "Runs Monday - Friday, 6:30 a.m. - 11 p.m., every 20 minutes" in downtown
    west = line_with(text, "Tempe - West Valley")
    assert "Stops Apache Boulevard and Rural Road, Fletcher Library" in west
    assert "every 30 minutes" in west
    poly = line_with(text, "Polytechnic Express")
    assert "Stops Tempe Transit Center, Sutton Hall" in poly
    assert "Runs Saturday - Sunday, 8 a.m. - 8 p.m., hourly" in poly
    assert "https://" not in text
    assert "|" in text and "| ---" not in text


def test_jobs_keep_title_department_pay_and_closing_date_together() -> None:
    text = extract_jobs(page("jobs.md"))
    assistant = line_with(text, "Student Office Assistant")
    assert "Department University Libraries" in assistant
    assert "Pay $15.50 per hour" in assistant
    assert "Closes September 19, 2025" in assistant
    guide = line_with(text, "Campus Tour Guide")
    assert "Department Admission Services" in guide
    assert "Pay $16.00 per hour" in guide
    assert "Closes October 3, 2025" in guide
    tech = line_with(text, "Lab Technician I")
    assert "Department School of Life Sciences" in tech
    assert "Pay $17.25 per hour" in tech
    assert "Closes September 26, 2025" in tech
    lifeguard = line_with(text, "Rec Center Lifeguard")
    assert "Closes Open until filled" in lifeguard
    assert "https://" not in text
    assert "](" not in text


def test_sports_keep_fixture_opponent_date_and_venue_together() -> None:
    text = extract_sports(page("sports.md"))
    texas_tech = line_with(text, "Texas Tech")
    assert texas_tech.startswith("Football")
    assert "Date September 13, 2025" in texas_tech
    assert "Location Mountain America Stadium, Tempe" in texas_tech
    baylor = line_with(text, "at Baylor")
    assert "Date September 20, 2025" in baylor
    assert "Location McLane Stadium, Waco" in baylor
    soccer = line_with(text, "Women's Soccer vs Utah")
    assert "Date September 12, 2025" in soccer
    assert "Location Sun Devil Soccer Stadium, Tempe" in soccer
    basketball = line_with(text, "Men's Basketball Exhibition")
    assert "Opponent Northern Arizona" in basketball
    assert "Date October 24, 2025" in basketball
    assert "Location Desert Financial Arena, Tempe" in basketball
    assert "https://" not in text
    assert "| ---" not in text


@pytest.mark.parametrize("key", sorted(SOURCES))
def test_registered_source_carries_its_extractor(key: str) -> None:
    assert callable(SOURCES[key].extractor)


@pytest.mark.parametrize(
    "extractor",
    [
        extract_hours,
        extract_events,
        extract_courses,
        extract_clubs,
        extract_scholarships,
        extract_news,
        extract_shuttles,
        extract_jobs,
        extract_sports,
    ],
    ids=[
        "hours",
        "events",
        "courses",
        "clubs",
        "scholarships",
        "news",
        "shuttles",
        "jobs",
        "sports",
    ],
)
def test_extractors_fall_back_to_the_raw_body_when_the_fetcher_gave_no_text(extractor) -> None:
    html = b"<html><body><main><p>CSE 110: Principles of Programming (3)</p></main></body></html>"
    fetched = Fetched(
        url="https://example.invalid/page", status=200, body=html, content_type="text/html"
    )
    assert "Principles of Programming" in extractor(fetched)


def test_an_extractor_that_returns_nothing_yields_no_chunks_for_the_pipeline_guard() -> None:
    empty = Fetched(
        url="https://example.invalid/empty",
        status=200,
        body=b"",
        content_type="text/markdown",
        text="   \n\n",
    )
    assert chunk_text(extract_hours(empty)) == []
