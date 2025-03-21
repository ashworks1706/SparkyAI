from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from datetime import datetime
import logging

class ASUCourseScraper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        chrome_options = Options()
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        self.driver = webdriver.Chrome(options=chrome_options)
        self.text_content = []

    def handle_cookie(self, driver):
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "cookie-close-button"))
            )
            cookie_close_button = driver.find_element(By.ID, "cookie-close-button")
            cookie_close_button.click()
            self.logger.info("Cookie popup handled")
        except:
            self.logger.info("No cookie popup found")

    def scrape_catalog_apps_asu_edu(self, url):
        self.logger.info(f"Scraping URL: {url}")
        self.driver.get(url)
        self.handle_cookie(self.driver)
        
        try:
            course_elements = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "class-accordion"))
            )
            self.logger.info(f"Found {len(course_elements)} course elements.")
        except Exception as e:
            self.logger.error(f"Failed to find course elements: {e}")
            return

        detailed_courses = []
        
        for course in course_elements[:7]:
            try:
                course_title_element = course.find_element(By.CSS_SELECTOR, ".course .bold-hyperlink")
                course_title = course_title_element.text
                self.logger.info(f"Processing course: {course_title}")
                
                # Use JavaScript click to handle potential interception
                self.driver.execute_script("arguments[0].click();", course_title_element)
                self.logger.info("\nSuccessfully clicked on the course")

                # Wait for dropdown to load
                details_element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, "class-details"))
                )
                
                # Extract additional details
                course_info = {
                    'title': course_title,
                }
                try:
                    course_info['description'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Course Description')]/following-sibling::p").text
                except:
                    course_info['description'] = 'N/A'
                    self.logger.info(f"Could not find course description for {course_title}")
                try:
                    course_info['enrollment_requirements'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Enrollment Requirements')]/following-sibling::p").text
                except:
                    course_info['enrollment_requirements'] = 'N/A'
                    self.logger.info(f"Could not find enrollment requirements for {course_title}")
                try:
                    location_element = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Location')]/following-sibling::p")
                    location_link = location_element.find_element(By.TAG_NAME, "a")
                    course_info['location'] = location_link.text
                except Exception as e:
                    self.logger.info(f"Error in web_scrap course location : {e}")
                    course_info['location'] = 'N/A'
                try:
                    course_info['number'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Number')]/following-sibling::p").text
                except:
                    course_info['number'] = 'N/A'
                    self.logger.info(f"Could not find course number for {course_title}")
                try:
                    course_info['units'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Units')]/following-sibling::p").text
                except:
                    course_info['units'] = 'N/A'
                    self.logger.info(f"Could not find units for {course_title}")
                try:
                    course_info['dates'] = details_element.find_element(By.CLASS_NAME, "text-nowrap").text
                except:
                    course_info['dates'] = 'N/A'
                    self.logger.info(f"Could not find dates for {course_title}")
                try:
                    course_info['offered_by'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Offered By')]/following-sibling::p").text
                except:
                    course_info['offered_by'] = 'N/A'
                    self.logger.info(f"Could not find who offered the course for {course_title}")
                try:
                    course_info['repeatable_for_credit'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Repeatable for credit')]/following-sibling::p").text
                except:
                    course_info['repeatable_for_credit'] = 'N/A'
                    self.logger.info(f"Could not find if repeatable for credit for {course_title}")
                try:
                    course_info['component'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Component')]/following-sibling::p").text
                except:
                    course_info['component'] = 'N/A'
                    self.logger.info(f"Could not find component for {course_title}")
                try:
                    course_info['last_day_to_enroll'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Last day to enroll')]/following-sibling::p").text
                except:
                    course_info['last_day_to_enroll'] = 'N/A'
                    self.logger.info(f"Could not find last day to enroll for {course_title}")
                try:
                    course_info['drop_deadline'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Drop deadline')]/following-sibling::p").text
                except:
                    course_info['drop_deadline'] = 'N/A'
                    self.logger.info(f"Could not find drop deadline for {course_title}")
                try:
                    course_info['course_withdrawal_deadline'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Course withdrawal deadline')]/following-sibling::p").text
                except:
                    course_info['course_withdrawal_deadline'] = 'N/A'
                    self.logger.info(f"Could not find course withdrawal deadline for {course_title}")
                try:
                    course_info['consent'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Consent')]/following-sibling::p").text
                except:
                    course_info['consent'] = 'N/A'
                    self.logger.info(f"Could not find consent information for {course_title}")
                try:
                    course_info['course_notes'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Course Notes')]/following-sibling::p").text
                except:
                    course_info['course_notes'] = 'N/A'
                    self.logger.info(f"Could not find course notes for {course_title}")
                try:
                    course_info['fees'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Fees')]/following-sibling::p").text
                except:
                    course_info['fees'] = 'N/A'
                    self.logger.info(f"Could not find fees for {course_title}")
                    
                try:
                    course_info['instructor'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Instructor')]/following-sibling::a").text
                except:
                    course_info['instructor'] = 'N/A'
                    self.logger.info(f"Could not find instructor for {course_title}")
                
                
                # Extract reserved seats information
                try:
                    reserved_seats_table = details_element.find_element(By.CLASS_NAME, "reserved-seats")
                    reserved_groups = []
                    rows = reserved_seats_table.find_elements(By.TAG_NAME, "tr")[1:-1]  # Skip header and last row
                    for row in rows:
                        cols = row.find_elements(By.TAG_NAME, "td")
                        reserved_groups.append({
                            'group': cols[0].text,
                            'available_seats': cols[1].text,
                            'students_enrolled': cols[2].text,
                            'total_seats_reserved': cols[3].text,
                            'reserved_until': cols[4].text
                        })
                    course_info['reserved_seats'] = reserved_groups
                except:
                    course_info['reserved_seats'] = []
                    self.logger.info(f"Could not find reserved seats for {course_title}")
                
                detailed_courses.append(course_info)
                
            except Exception as e:
                self.logger.error(f"Error processing course {e}")

                
            formatted_courses = []
            for course in detailed_courses:
                course_string = f"Title: {course['title']}\n"
                course_string += f"Description: {course['description']}\n"
                course_string += f"Enrollment Requirements: {course['enrollment_requirements']}\n"                    
                course_string += f"Instructor: {course['instructor']}\n"
                course_string += f"Location: {course['location']}\n"
                course_string += f"Course Number: {course['number']}\n"
                course_string += f"Units: {course['units']}\n"
                course_string += f"Dates: {course['dates']}\n"
                course_string += f"Offered By: {course['offered_by']}\n"
                course_string += f"Repeatable for Credit: {course['repeatable_for_credit']}\n"
                course_string += f"Component: {course['component']}\n"
                course_string += f"Last Day to Enroll: {course['last_day_to_enroll']}\n"
                course_string += f"Drop Deadline: {course['drop_deadline']}\n"
                course_string += f"Course Withdrawal Deadline: {course['course_withdrawal_deadline']}\n"
                course_string += f"Consent: {course['consent']}\n"
                course_string += f"Course Notes: {course['course_notes']}\n"
                course_string += f"Fees: {course['fees']}\n"

                # Add reserved seats information
                if course.get('reserved_seats'):
                    course_string += "Reserved Seats:\n"
                    for group in course['reserved_seats']:
                        course_string += f"- Group: {group['group']}\n"
                        course_string += f"  Available Seats: {group['available_seats']}\n"
                        course_string += f"  Students Enrolled: {group['students_enrolled']}\n"
                        course_string += f"  Total Reserved Seats: {group['total_seats_reserved']}\n"
                        course_string += f"  Reserved Until: {group['reserved_until']}\n"
                self.text_content.append({
                        'content': course_string,
                        'metadata': {
                            'url': url,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }
                    })
                self.logger.info(f"Appended {self.text_content[-1]}")
                formatted_courses.append(course_string)
    
    def scrape_google_search(self, url):
        self.logger.info(f"Initiating google search scrape for {url}")
        self.driver.get(url)
        self.scrape_catalog_apps_asu_edu(url)

    def close(self):
        self.logger.info("Closing the driver")
        self.driver.quit()

if __name__ == '__main__':
    scraper = ASUCourseScraper()
    try:
        scraper.scrape_google_search("https://search.lib.asu.edu/discovery/search?query=any,contains,dale%20carnegie&tab=Everything&search_scope=MyInst_and_CI&sortby=date_d&vid=01ASU_INST:01ASU&facet=frbrgroupid,include,9049087968550056342&offset=0")
    finally:
        scraper.close()