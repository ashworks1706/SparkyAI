import subprocess
import platform
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import time
import unittest

class WebScraper:
    def __init__(self):
        self.visited_urls = set()
        self.text_content = []
        self.logged_in_driver = None
        self.chrome_options = Options()
        # if you want to start chrome supressed enable this comment
        # self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--window-size=1920,1080')
        self.chrome_options.add_argument('--ignore-certificate-errors')
        self.chrome_options.add_argument('--disable-extensions')
        self.chrome_options.add_argument('--no-first-run')
        self.chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # Prevent detection as a bot


        # print("Enter Chrome binary location")
        print('/usr/bin/google-chrome-stable # Standard Linux path')
        print('/mnt/c/Program Files/Google/Chrome/Application/chrome.exe # Standard WSL path')

        if platform.system() == 'Linux':
            self.chrome_options.binary_location = '/usr/bin/google-chrome-stable'  # Standard Linux path
        elif 'microsoft' in platform.uname().release.lower():  # WSL detection
            self.chrome_options.binary_location = '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe'

        try:
            # Get Chrome version (3rd element in output)
            chrome_out = subprocess.check_output(
                [self.chrome_options.binary_location, '--version']
            ).decode().strip()
            chrome_version = chrome_out.split()[2]

            # Get Chromedriver version (2nd element in output)
            driver_out = subprocess.check_output(
                ['chromedriver', '--version']
            ).decode().strip()
            driver_version = driver_out.split()[1]

            print(f"Chrome: {chrome_version}, Chromedriver: {driver_version}")

            if chrome_version != driver_version:
                raise RuntimeError(f"Mismatch: Chrome {chrome_version} vs Driver {driver_version}")

        except IndexError as e:
            print(f"Version parsing failed. Raw output:\nChrome: {chrome_out}\nDriver: {driver_out}")
            raise

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.popup = False

        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.chrome_options)

    def handle_feedback_popup(self, driver):
        if self.popup:
            return

        try:
            print("\nHandling feedback popup")
            # Wait for the popup to be present
            popup = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, "fsrDeclineButton"))
            )

            # Click the "No thanks" button
            popup.click()
            print("\nFeedback popup clicked")
            # Optional: Wait for popup to disappear
            WebDriverWait(driver, 5).until(
                EC.invisibility_of_element_located((By.ID, "fsrFullScreenContainer"))
            )

            self.popup = True
        except Exception as e:
            print("Error handling feedback popup")


    def scrape_asu_library(self, url):
        if 'search.lib.asu.edu' not in url:
            print(f"URL {url} is not an ASU library search URL. Skipping.")
            return False

        self.driver.get(url)
        time.sleep(1)
        book_results = []
        self.handle_feedback_popup(self.driver)

        try:
            # Find and click on the first book title link
            first_book_link = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".result-item-text div"))
            )
            first_book_link.click()
            print("\nBook Title Clicked")

            # Wait for book details to be present
            print("\nBook Details fetched")
            for _ in range(3):
                book_details = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.full-view-inner-container.flex"))
                )
                self.handle_feedback_popup(self.driver)
                
                # Extract book title
                author_view = self.driver.find_element(By.CSS_SELECTOR,
                                                        "div.result-item-text.layout-fill.layout-column.flex")
                print("\nAuthors fetched")

                title = author_view.find_element(By.CSS_SELECTOR, "h3.item-title").text.strip()
                print("\nBook Title fetched")

                # Extract Authors
                authors = []

                try:
                    author_div = author_view.find_element(By.XPATH,
                                                            "//div[contains(@class, 'item-detail') and contains(@class, 'reduce-lines-display')]")

                    # Find all author elements within this div
                    author_elements = author_div.find_elements(By.CSS_SELECTOR,
                                                                "span[data-field-selector='creator'], span[data-field-selector='contributor']")

                    if len(author_elements) > 0:
                        for element in author_elements:
                            author_text = element.text.strip()
                            if author_text and author_text not in authors:
                                authors.append(author_text)
                    else:
                        author_div = book_details.find_element(By.XPATH, "//div[.//span[@title='Author']]")

                        author_elements = author_div.find_elements(By.CSS_SELECTOR,
                                                                    "a span[ng-bind-html='$ctrl.highlightedText']")

                        if not author_elements:
                            author_elements = book_details.find_elements(By.XPATH,
                                                                            "//div[contains(@class, 'item-details-element')]//a//span[contains(@ng-bind-html, '$ctrl.highlightedText')]")
                        if len(author_elements) > 0:
                            for element in author_elements:
                                author_text = element.text.strip()
                                if author_text and author_text not in authors:
                                    authors.append(author_text)
                    print("\nAuthors fetched")

                except Exception as e:
                    authors = 'N/A'

                try:
                    publisher = book_details.find_element(By.CSS_SELECTOR,
                                                            "span[data-field-selector='publisher']").text.strip()
                    print("\nPublisher fetched")
                except:
                    print("\nNo Publisher found")
                    publisher = "N/A"

                # Extract publication year
                try:
                    year = book_details.find_element(By.CSS_SELECTOR,
                                                        "span[data-field-selector='creationdate']").text.strip()
                except:
                    print("\nNo Book details found")
                    year = "N/A"

                # Extract availability
                try:
                    location_element = book_details.find_element(By.CSS_SELECTOR, "h6.md-title")
                    availability = location_element.text.strip()
                    print("\nAvailability found with first method")

                except Exception as e:
                    # Find the first link in the exception block
                    location_element = book_details.find_elements(By.CSS_SELECTOR,
                                                                    "a.item-title.md-primoExplore-theme")
                    try:
                        
                        if isinstance(location_element, list):
                            availability = location_element[0].get_attribute('href')
                        else:
                            availability = location_element.get_attribute('href')
                        print("\nAvailability found with second method")

                        if availability is None:
                            location_element = book_details.find_elements(By.CSS_SELECTOR,
                                                                            "h6.md-title ng-binding zero-margin")
                            availability = location_element.text.strip()
                            print("\nAvailablility found with third method")
                    except:
                        print("\nNo availability found")
                        availability = "N/A"

                try:
                    # Use more flexible locator strategies
                    links = self.driver.find_elements(By.XPATH, "//a[contains(@ui-sref, 'sourceRecord')]")

                    if isinstance(links, list) and len(links) > 0:
                        link = links[0].get_attribute('href')
                        print("\nFetched Link")
                    else:
                        link = 'N/A'
                        print("\nNo link Found")
                except Exception as e:
                    print("\nNo link Found")
                    link = 'N/A'

                # Compile book result
                book_result = {
                    "title": title,
                    "authors": authors,
                    "publisher": publisher,
                    "year": year,
                    "availability": availability,
                    "link": link
                }

                book_results.append(book_result)

                try:
                    next_button = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(@ng-click, '$ctrl.getNextRecord()')]"))
                    )
                    self.driver.execute_script("arguments[0].click();", next_button)
                    
                    time.sleep(3)

                    print("\nClicked next button")

                    self.handle_feedback_popup(self.driver)

                except Exception as e:
                    print(f"Failed to click next button: {e}")

            if len(book_results) == 0:
                return False

            for book in book_results:
                book_string = f"Title: {book['title']}\n"
                book_string += f"Authors: {', '.join(book['authors']) if book['authors'] else 'N/A'}\n"
                book_string += f"Publisher: {book['publisher']}\n"
                book_string += f"Publication Year: {book['year']}\n"
                book_string += f"Availability: {book['availability']}\n"
                book_string += f"Link: {book['link']}\n"

                self.text_content.append({
                    'content': book_string,
                    'metadata': {
                        'url': book['link'],
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    }
                })
                print("\nAppended book details: %s" % self.text_content[-1])

            return True

        except Exception as e:
            print(f"\nFailed to scrape book details: {e}")
            return False

    def close(self):
        """Closes the webdriver."""
        if self.driver:
            self.driver.quit()
            print("Webdriver closed.")


    def test_scrape_asu_library_valid_url(self):
        # Test with a valid URL
        url = "https://search.lib.asu.edu/discovery/search?query=any,contains,dale%20carnegie&tab=Everything&search_scope=MyInst_and_CI&sortby=date_d&vid=01ASU_INST:01ASU&facet=frbrgroupid,include,9049087968550056342&lang=en&offset=0"
        result = self.scrape_asu_library(url)

        # Assert that the method returns True for valid URLs
        print(result)



if __name__ == '__main__':
    WebScraper().test_scrape_asu_library_valid_url() 