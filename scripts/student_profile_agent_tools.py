from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import re
from selenium.webdriver.common.action_chains import ActionChains

class student_profile_agent_tools:

    def __init__(self, firestore, utils, logger):
        self.firestore = firestore
        self.utils = utils
        self.visited_urls = set()
        self.max_depth = 2
        self.max_links_per_page = 3
        self.service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service)

    async def handle_cookie(self, driver):
        try:
            self.logger.info("Waiting for cookie popup...")
            agree_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[aria-label="dismiss cookie message"]'))
            )
            driver.execute_script("arguments[0].click();", agree_button)
            self.logger.info("Cookie popup found and dismissed!")
        except Exception as e:
            self.logger.info(f"No cookie popup found or failed to dismiss: {e}")

    def quit(self):
        self.driver.quit()

    async def get_taken_classes(self, specified_term):
        try:
            self.logger.info("Launching driver for taken classes")
            driver.get("https://webapp4.asu.edu/eadvisor/")  
            time.sleep(5)  
            handle_cookie(driver)
        except Exception as e:
            self.logger.info(f"Error launching driver for taken classes: {e}")
        
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        course_list = []

        try:
            self.logger.info("Starting scraping for taken classes")
            for row in soup.find_all("div", class_="term_row"):
            courses_div = row.find("div", class_="courses")
            if not courses_div:
                continue

            for course_div in courses_div.find_all("div", class_="course"):
                course = course_div.get_text(strip=True)

                grade_div = course_div.find_next_sibling("div", class_="grade")
                term_div = course_div.find_next_sibling("div", class_="taken")

                if not grade_div or not term_div:
                    continue
                
                grade = grade_div.get_text(strip=True)
                term = term_div.get_text(strip=True) 

                if (course, grade, term) not in course_list and course != "" and grade != "" and term != "" and '.' not in course_list and (specified_term == "" or term == specified_term):
                    course_list.append((course, grade, term))

        except Exception as e:
            self.logger.info(f"Error in scraping taken classes: {e}")
        
        return course_list

    async def get_schedule(self, specified_term):
        try:
            self.logger("Launching driver for schedule")
            driver.get("https://webapp4.asu.edu/myasu/student/schedule")  
            time.sleep(5)
        except Exception as e:
            self.logger.info(f"Error launching driver for schedule: {e}")

        try:
            self.logger("Opening dropdown button menu")
            dropdown_button = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "schedule_dropdown")))
        except Exception as e:
            self.logger.info(f"Error with dropdown button: {e}")
            
        driver.execute_script("arguments[0].click();", dropdown_button)

        try:
            self.logger("Finding all term links")
            term_links = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.myasu-tippy-option")))
        except Exception as e:
            self.logger.info(f"Error with finding all term links: {e}")

        try:
            for link in term_links:
                if specified_term in link.text and specified_term != "":
                    driver.execute_script("arguments[0].click();", link)
                    break
        except Exception as e:
            self.logger.info(f"Did not fink right term: {e}")

        time.sleep(5)

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        schedule = []

        try:
            self.logger.info("Starting scraping for schedule")
            for row in soup.find_all("div", class_ = "class-content-container"):
                course_num_div = row.find("div", class_ = "class-number-column")
                course_num = course_num_div.get_text(" ", strip = True) if course_num_div else "N/A"
                
                course_title_div = row.find("span", class_ = "class-title")
                course_title = re.sub(r'\s+', ' ', course_title_div.get_text()).strip() if course_title_div else "N/A"

                units_div = row.find("div", class_ = "units-column")
                units = (units_div.find("label")).get_text(" ", strip = True) if units_div else "N/A"
                
                instructor_div = row.find("div", class_ = "instructors-column")
                instructor = re.sub(r'\s+', ' ', (instructor_div.find("div", class_ = "class-content-inner-cell")).get_text(separator = " ", strip = True)).strip() if instructor_div else "N/A"
                
                course_days_div = row.find("div", attrs = {"data-label": "Days"})
                course_days = (course_days_div.find("div", class_ = "class-content-inner-cell")).get_text(separator=" ", strip = True) if course_days_div else "N/A"
                course_days = course_days if course_days and course_days.strip() != "" else "N/A"
                
                course_time_div = row.find("div", attrs = {"data-label": "Times"})
                course_times = (course_time_div.find("div", class_ = "class-content-inner-cell")).get_text(separator = " ", strip = True) if course_time_div else "N/A"
                course_times = course_times if course_times and course_times.strip() != "" else "N/A"
                
                course_dates_div = row.find("div", class_ = "dates-column")
                course_dates = (course_dates_div.find("div", class_ = "class-content-inner-cell")).get_text(separator = " ", strip = True) if course_dates_div else "N/A"
                
                course_location_div = row.find("div", attrs = {"data-label": "Location"})
                course_location = (course_location_div.find("div", class_="class-content-inner-cell")).get_text(separator=" ", strip=True) if course_location_div else "N/A"    
                
                schedule.append((course_num, course_title, units, instructor, course_days, course_times, course_dates, course_location))
        except Exception as e:
            self.logger.info(f"Error in scraping for schedule: {e}")

        return schedule
    
    async def get_scholarships(self):
        try:
            self.logger.info("Launching driver for scholarships")
            driver.get("https://webapp4.asu.edu/myasu/student/finances")
            time.sleep(10)
        except Exception as e:
            self.logger.info(f"Error in getting scholarships: {e}")

        html = driver.page_source

        soup = BeautifulSoup(html, "html.parser")

        awards = []

        try:
            self.logger.info("Starting scraping for scholarships")
            for container in soup.find_all("div", class_ = "box-padding"):
                term_th = container.find("th", attrs = {"align": "left"})
                if not term_th:
                    continue

                term = term_th.get_text(" ", strip = True)

                table = container.find("table")
                if not table:
                    continue

                body = container.find("tbody")

                for tr in body.find_all("tr"):
                    if not tr.has_attr("class"):
                        award_title_span = tr.find("td", class_ = "award-title")
                        award_amount_td = tr.find("td", attrs = {"align": "right"})
                
                        if award_title_span and award_amount_td and term_th:
                            award_title = award_title_span.get_text(" ", strip = True)
                            award_amount = award_amount_td.get_text(" ", strip = True)
                                
                            awards.append((award_title, award_amount, term))        
        except Exception as e:
            self.logger.info(f"Error scraping scholarships: {e}")

        return awards

    async def get_current_charges(self):
        try:
            self.logger.info("Launching driver for charges")
            driver.get("https://webapp4.asu.edu/myasu/student/finances")
            time.sleep(10)
        except Exception as e:
            self.logger.info(f"Error launching driver for charges: {e}")

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        charges = []

        try:
            self.logger.info("Starting scraping for charges")
            container = soup.find("div", attrs = {"id": "charge-summary-current-body"})
            table = container.find("table")
            rows = table.find("tbody")

            for tr in rows.find_all("tr"):
                tds = tr.find_all("td")

                # Only process rows with enough data
                if len(tds) >= 3:
                    term = tds[0].get_text(strip = True)
                    due_date = tds[1].get_text(strip = True)
                    description = tds[2].get_text(strip = True)
                    amount = tds[3].get_text(strip = True) if len(tds) > 3 else ""

                    charges.append((term, due_date, description, amount))
        except Exception as e:
            self.logger.info(f"Error scraping for charges: {e}")
        
        return charges
    
    async def get_advisor_info(self):
        try:
            self.logger.info("Launching driver for advisor info")
            driver.get("https://webapp4.asu.edu/myasu/student")
            time.sleep(10)
        except Exception as e:
            self.logger.info(f"Error scraping for advisor info: {e}")

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        advisor_info = []

        try:
            self.logger.info("Launching info for advisor info")
            advisor_label = soup.find("div", class_ = "table-cell", string = "Your Advisor:")
            advisor_name = advisor_label.find_next_sibling("div", class_ = "table-cell").get_text(strip = True)

            contact_label = soup.find("div", class_ = "table-cell", string = "Contact:")
            contact_div = contact_label.find_next_sibling("div", class_ = "table-cell")

            phone = None
            email = None

            if contact_div:
                for link in contact_div.find_all("a"):
                    href = link.get("href", "")
                    text = link.get_text(strip = True)
                    if "tel:" in href:
                        phone = text
                    elif "mailto:" in href:
                        email = text

            advisor_info.append((advisor_name, phone, email))
        except Exception as e:
            self.logger.info(f"Error scraping for advisor info: {e}")
    
        return advisor_info
