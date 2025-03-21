from utils.common_imports import *
import subprocess
import platform

class ASUWebScraper:
    def __init__(self,discord_state,utils,logger):
        self.discord_client = discord_state.get('discord_client')
        self.visited_urls = set()
        self.utils = utils
        self.text_content = []
        self.logged_in_driver= None
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

        self.logger= logger
        
        # logger.info("Enter Chrome binary location")
        logger.info('/usr/bin/google-chrome-stable # Standard Linux path')
        logger.info('/mnt/c/Program Files/Google/Chrome/Application/chrome.exe # Standard WSL path')
        
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
            
            logger.info(f"Chrome: {chrome_version}, Chromedriver: {driver_version}")
            
            if chrome_version != driver_version:
                raise RuntimeError(f"Mismatch: Chrome {chrome_version} vs Driver {driver_version}")
                
        except IndexError as e:
            logger.error(f"Version parsing failed. Raw output:\nChrome: {chrome_out}\nDriver: {driver_out}")
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
        
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=self.chrome_options)

    
    
    def handle_feedback_popup(self,driver):
        if self.popup:
            pass
        else:
            try:
                self.logger.info("\nHandling feedback popup")
                # Wait for the popup to be present
                popup = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "fsrDeclineButton"))
                )
                
                # Click the "No thanks" button
                popup.click()
                self.logger.info("\nFeedback popup clicked")
                # Optional: Wait for popup to disappear
                WebDriverWait(driver, 5).until(
                    EC.invisibility_of_element_located((By.ID, "fsrFullScreenContainer"))
                )
                
                self.popup = True
            except Exception as e:

                pass
    
    def handle_cookie(self,driver):
        if self.popup: 
            pass
        else:
            try:
                self.logger.info("\nHandling feedback popup")
                cookie_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "rcc-confirm-button"))
                )
                cookie_button.click()
                self.logger.info("\nSuccessfully clciked on cookie button")
            except:
                pass

    
    async def __login__(self, username, password):
        try:
            # Navigate to Handshake login page
            self.driver.get("https://asu.joinhandshake.com/login?ref=app-domain")
            
            # Wait for page to load
            wait = WebDriverWait(self.driver, 10)
            
            # Find and click "Sign in with your email address" button
            email_signin_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//a[@data-bind='click: toggle']"))
            )
            email_signin_button.click()
            
            # Enter email address
            email_input = wait.until(
                EC.presence_of_element_located((By.ID, "non-sso-email-address"))
            )
            email_input.send_keys(username)
            
            # Click "Next" button
            next_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[@type='submit' and contains(@class, 'button')]"))
            )
            next_button.click()
            
            # Click "Or log in using your Handshake credentials"
            alternate_login = wait.until(
                EC.element_to_be_clickable((By.CLASS_NAME, "alternate-login-link"))
            )
            alternate_login.click()
            
            # Enter password
            password_input = wait.until(
                EC.presence_of_element_located((By.ID, "password"))
            )
            password_input.send_keys(password)
            
            # Submit login
            submit_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='submit' and contains(@class, 'button default-focus')]"))
            )
            submit_button.click()
            
            
            # Store the logged-in driver state
            self.logged_in_driver = self.driver
            
            return True
        
        except Exception as e:
            self.logger.error(f"Login failed: {str(e)}")
            return False
        
    
    
    
    async def scrape_content(self, url: str, query_type: str = None, max_retries: int = 3, selenium :bool = False, optional_query:str=None) -> bool:
        """Scrape content using Jina.ai"""
        
        self.logger.info(f"Scraping url : {url} ")
        self.logger.info(f"query_type : {query_type} ")
        self.logger.info(f"max_retries : {max_retries} ")
        self.logger.info(f"selenium required : {selenium} ")
        self.logger.info(f"optional query : {optional_query} ")
        
        await self.utils.update_text("Understanding Results...")

        if isinstance(url, dict):
            url = url.get('url', '')
        
        # Ensure url is a string and not empty
        if not isinstance(url, str) or not url:
            self.logger.error(f"Invalid URL: {url}")
            return False
        
        # Validate and format the URL
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme:
                url = f"https://{url}"
                parsed_url = urlparse(url)
            if not parsed_url.netloc:
                self.logger.error(f"Malformed URL: {url}")
                return False
        except Exception as e:
            self.logger.error(f"Error parsing URL: {url}, Exception: {str(e)}")
            return False
        
        if url in self.visited_urls:
            return False
        
        
        self.visited_urls.add(url)
        
        if not selenium:
            for attempt in range(max_retries):
                try:
                    loader = UnstructuredURLLoader(urls=[url])
                    documents = loader.load()
                    
                    if documents and documents[0].page_content and len(documents[0].page_content.strip()) > 50 and not "requires javascript to be enabled" in documents[0].page_content.lower() and not "supported browser" in documents[0].page_content.lower() and not "captcha" in documents[0].page_content.lower():
                        self.text_content.append({
                                'content': documents[0].page_content,
                                'metadata': {
                                    'url': url,
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'title': documents[0].metadata.get('title', ''),
                                    'description': documents[0].metadata.get('description', ''),
                                }
                            })
                        return True
                    else:
                        self.logger.info("Langchain method failed")
                except Exception as e:
                    self.logger.error(f"Error fetching content from {url}: {str(e)}")
                    await asyncio.sleep(8) 
                    continue  
                else:
                    jina_url = f"https://r.jina.ai/{url}"
                    response = requests.get(jina_url, headers=self.headers, timeout=30)
                    response.raise_for_status()
                    
                    text = response.text
                    self.logger.info(f"Raw text response for https://r.jina.ai/{url}\n{text}")
                    
                    if "LOADING..." in text.upper() or "requires javascript to be enabled" in text.lower() or "supported browser" in text.lower() or "captcha" in text.lower():
                        self.logger.warning(f"LOADING response detected for {url}. Retry attempt {attempt + 1}")
                        await asyncio.sleep(8)  # Wait before retrying
                        continue
                    self.logger.info("Scrarping successfull")
                    if text and len(text.strip()) > 50:
                        self.text_content.append({
                            'content': text,
                            'metadata': {
                                'url': url,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            }
                        })
                        
                        self.logger.info(self.text_content[-1])
                        return True
                    
            return False
        
        elif 'postings' in url:

                
            # Navigate to Handshake job postings
            self.logged_in_self.driver.get(url)
            
            
            # Wait for page to load
            wait = WebDriverWait(self.logged_in_driver, 10)
            
            # Parse optional query parameters
            if optional_query:
                query_params = parse_qs(optional_query)
                
                
                # Search Bar Query
                if 'search_bar_query' in query_params:
                    search_input = wait.until(
                        EC.presence_of_element_located((By.XPATH, "//input[@role='combobox']"))
                    )
                    search_input.send_keys(query_params['search_bar_query'][0])
                    self.logger.info("\nSuccessfully entered search_bar_query")
                
                
                if 'job_location' in query_params:
                    location_button = wait.until(
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'style__pill___3uHDM') and .//span[text()='Location']]"))
                    )
                    location_button.click()

                    location_input = wait.until(
                        EC.presence_of_element_located((By.ID, "locations-filter"))
                    )
                    
                    # Remove list brackets and use the first element directly
                    job_location = query_params['job_location'][0]
                    
                    location_input.clear()
                    location_input.send_keys(job_location)
                    self.logger.info("\n Successfully entered job_location")

                    try:
                        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "mapbox-autocomplete-results")))
                        time.sleep(1)

                        first_location = wait.until(
                            EC.element_to_be_clickable((
                            By.XPATH, 
                            f"//div[contains(@class, 'mapbox-autocomplete-results')]//label[contains(text(), '{job_location}')]"
                            )))
                        
                        first_location.click()
                        
                        self.logger.info(f"Selected location: {job_location}")
                    except Exception as e:
                        self.logger.error(f"Error job location: {e}")

                all_filters_button = wait.until(
                    EC.element_to_be_clickable((
                        By.XPATH, 
                        "//button[contains(@class, 'style__pill___3uHDM') and .//span[text()='All filters']]"
                    ))
                )
                all_filters_button.click()
                
                self.logger.info("\nClicked on all filters")
                                    
                
                if 'job_type' in query_params:
                    # Get the specific job type from query parameters
                    job_type = query_params['job_type'][0]
                    
                    # Function to force click using JavaScript with multiple attempts
                    def force_click_element(driver, element, max_attempts=3):
                        for attempt in range(max_attempts):
                            self.logger.info("\nAttempting to force click")
                            try:
                                # Try different click methods
                                self.driver.execute_script("arguments[0].click();", element)
                                time.sleep(0.5)  # Short pause to allow for potential page changes
                                return True
                            except Exception:
                                # Try alternative click methods
                                try:
                                    element.click()
                                except Exception:
                                    # Last resort: move and click
                                    try:
                                        ActionChains(self.driver).move_to_element(element).click().perform()
                                    except Exception:
                                        continue
                        return False
                    
                    # Check if the job type is in the first level of buttons (Full-Time, Part-Time)
                    standard_job_types = ['Full-Time', 'Part-Time']
                    
                    if job_type in standard_job_types:
                        # Direct selection for standard job types
                        try:
                            job_type_button = wait.until(
                                EC.presence_of_element_located((
                                    By.XPATH,
                                    f"//button[contains(@class, 'style__pill___3uHDM') and .//div[@data-name='{job_type}' and @tabindex='-1']]"
                                ))
                            )
                            force_click_element(self.driver, job_type_button)
                            self.logger.info("\nSelect job type")
                        except Exception:
                            pass
                    else:
                        # For nested job types, click More button first
                        try:
                            more_button = wait.until(
                                EC.presence_of_element_located((
                                    By.XPATH, 
                                    "//button[contains(@class, 'style__pill___') and contains(text(), '+ More')]"
                                ))
                            )
                            force_click_element(self.driver, more_button)
                            self.logger.info("\nClicked more button")
                            
                            # Wait and force click the specific job type button from nested options
                            job_type_button = wait.until(
                                EC.presence_of_element_located((
                                    By.XPATH,
                                    f"//button[contains(@class, 'style__pill___3uHDM') and .//div[@data-name='{job_type}' and @tabindex='-1']]"
                                ))
                            )
                            force_click_element(self.driver, job_type_button)
                            self.logger.info("\nSelect Job type")
                        except Exception:
                            pass
                    
                
                
                # Wait for the Show results button to be clickable
                show_results_button = wait.until(
                    EC.element_to_be_clickable((
                        By.CLASS_NAME, 
                        "style__clickable___3a6Y8"
                    ))
                )

                # Optional: Add a small delay before clicking to ensure page is ready
                time.sleep(4)

                # Force click the Show results button using JavaScript
                self.driver.execute_script("arguments[0].click();", show_results_button)





                try:
                    # Wait for job cards to be present using data-hook
                    job_cards = wait.until(
                        EC.presence_of_all_elements_located(
                            (By.CSS_SELECTOR, "[data-hook='jobs-card']")
                        )
                    )
                    text_content = []  # Limit to top 3 jobs
                    
                    for job_card in job_cards[:3]:
                        full_job_link = job_card.get_attribute('href')

                        self.driver.execute_script("arguments[0].click();", job_card)
                        self.logger.info("\nClicked Job Card")
                        
                        # Wait for preview panel to load using data-hook
                        wait.until(
                            EC.presence_of_element_located(
                                (By.CSS_SELECTOR, "[data-hook='details-pane']")
                            )
                        )
                        
                        # Find 'More' button using a more robust selector
                        more_button = wait.until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.view-more-button"))
                        )
                        self.driver.execute_script("arguments[0].click();", more_button)
                        self.logger.info("\nClicked 'More' button")

                        
                        time.sleep(1)
                        h = html2text.HTML2Text()
                        time.sleep(2)
                        
                        job_preview_html = self.driver.find_element(By.CLASS_NAME, "style__details-padding___Y_KHb")
                        
                        soup = BeautifulSoup(job_preview_html.get_attribute('outerHTML'), 'html.parser')
    
                        # Find and remove the specific div with the class
                        unwanted_div = soup.find('div', class_='sc-gwVtdH fXuOWU')
                        if unwanted_div:
                            unwanted_div.decompose()
                        
                        unwanted_div = soup.find('div', class_='sc-dkdNSM eNTbTl')
                        if unwanted_div:
                            unwanted_div.decompose()
                        unwanted_div = soup.find('div', class_='sc-jEYHeb hSVHZy')
                        if unwanted_div:
                            unwanted_div.decompose()
                        unwanted_div = soup.find('div', class_='sc-VJPgA bRBKUF')
                        if unwanted_div:
                            unwanted_div.decompose()
                        

                        markdown_content = h.handle(str(soup))
                        
                        # remove image links
                        markdown_content = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_content)
    
                        # Remove hyperlinks
                        markdown_content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', markdown_content)

                        markdown_content = markdown_content.replace('\n',' ')
                        
                        markdown_content = markdown_content.replace('[',' ')
                        
                        markdown_content = markdown_content.replace(']',' ')
                        
                        markdown_content = markdown_content.replace('/',' ')
                        
                        self.text_content.append({
                            'content': markdown_content,
                            'metadata': {
                                'url': full_job_link,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            }
                        })
                except Exception as e:
                        self.logger.error(f"Error html to makrdown conversion :  {e}")
            
            return True
        
        elif 'catalog.apps.asu.edu' in url:
            self.driver.get(url)
            
            self.handle_cookie(self.driver)
            course_elements = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "class-accordion"))
            )

            detailed_courses = []
            
            for course in course_elements[:7]:
                try:
                    course_title_element = course.find_element(By.CSS_SELECTOR, ".course .bold-hyperlink")
                    course_title = course_title_element.text
                    
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
                    try:
                        course_info['enrollment_requirements'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Enrollment Requirements')]/following-sibling::p").text
                    except:
                        course_info['enrollment_requirements'] = 'N/A'
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
                    try:
                        course_info['units'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Units')]/following-sibling::p").text
                    except:
                        course_info['units'] = 'N/A'
                    try:
                        course_info['dates'] = details_element.find_element(By.CLASS_NAME, "text-nowrap").text
                    except:
                        course_info['dates'] = 'N/A'
                    try:
                        course_info['offered_by'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Offered By')]/following-sibling::p").text
                    except:
                        course_info['offered_by'] = 'N/A'
                    try:
                        course_info['repeatable_for_credit'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Repeatable for credit')]/following-sibling::p").text
                    except:
                        course_info['repeatable_for_credit'] = 'N/A'
                    try:
                        course_info['component'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Component')]/following-sibling::p").text
                    except:
                        course_info['component'] = 'N/A'
                    try:
                        course_info['last_day_to_enroll'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Last day to enroll')]/following-sibling::p").text
                    except:
                        course_info['last_day_to_enroll'] = 'N/A'
                    try:
                        course_info['drop_deadline'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Drop deadline')]/following-sibling::p").text
                    except:
                        course_info['drop_deadline'] = 'N/A'
                    try:
                        course_info['course_withdrawal_deadline'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Course withdrawal deadline')]/following-sibling::p").text
                    except:
                        course_info['course_withdrawal_deadline'] = 'N/A'
                    try:
                        course_info['consent'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Consent')]/following-sibling::p").text
                    except:
                        course_info['consent'] = 'N/A'
                    try:
                        course_info['course_notes'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Course Notes')]/following-sibling::p").text
                    except:
                        course_info['course_notes'] = 'N/A'
                    try:
                        course_info['fees'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Fees')]/following-sibling::p").text
                    except:
                        course_info['fees'] = 'N/A'
                        
                    try:
                        course_info['instructor'] = details_element.find_element(By.XPATH, ".//h5[contains(text(), 'Instructor')]/following-sibling::a").text
                    except:
                        course_info['instructor'] = 'N/A'
                    
                    
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
        
        elif 'search.lib.asu.edu' in url:
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
                
                time.sleep(2)
                
                if nested_book_link := self.driver.find_element(By.CSS_SELECTOR, ".result-item-text div"):
                    nested_book_link.click()
                    
                book_details = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.full-view-inner-container.flex"))
                )
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
     
        elif 'lib.asu.edu' in url:
            def extract_query_parameters(query):
                pattern = r'(\w+)=([^&]*)'
                matches = re.findall(pattern, query)
                parameters = [{param: value} for param, value in matches]
                return parameters

            # Classify the extracted parameters into lists
            library_names = []
            dates = []
            results = []

            # Extract parameters from the query string
            params = extract_query_parameters(optional_query)

            # Populate the lists based on parameter types
            for param in params:
                for key, value in param.items():
                    if key == 'library_names' and value != 'None':
                        library_names.append(value.replace("['", "").replace("']", ""))
                    if key == 'date' and value != 'None':
                        dates.append(value.replace("['","").replace("']",""))
            
            try:
                # Navigate to library hours page
                try:
                    self.self.driver.get(url)
                except Exception as e:
                    self.logger.error(f"Error navigating to URL: {e}")
                    return f"Error navigating to URL: {str(e)}"

                # Wait for page to load
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "s-lc-whw"))
                    )
                except Exception as e:
                    self.logger.error(f"Error waiting for page to load: {e}")
                    return f"Error waiting for page to load: {str(e)}"
                
                # Handle cookie popup
                try:
                    cookie_button = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.CLASS_NAME, "accept-btn"))
                    )
                    cookie_button.click()
                except Exception as e:
                    self.logger.warning(f"Error handling cookie popup: {e}")

                # Map library names to their row identifiers
                library_map = {
                    "Tempe Campus - Noble Library": "Noble Library",
                    "Tempe Campus - Hayden Library": "Hayden Library",
                    "Downtown Phoenix Campus - Fletcher Library": "Downtown campus Library",
                    "West Campus - Library": "Fletcher (West Valley)",
                    "Polytechnic Campus - Library": "Polytechnic"
                }


                # Process each library and date
                for library_name in library_names: 
                    
                    for date in dates:    
                        iterations = 0
                        is_date_present = False

                        while not is_date_present:
                            # Find all date headers in the thead
                            try:
                                date_headers = self.self.driver.find_elements(
                                    By.XPATH, "//thead/tr/th/span[@class='s-lc-whw-head-date']"
                                )
                            except Exception as e:
                                self.logger.error(f"Error finding date headers: {e}")
                                return f"Error finding date headers: {str(e)}"
                            
                            # Extract text from date headers
                            try:
                                header_dates = [header.text.strip() for header in date_headers]
                            except Exception as e:
                                self.logger.error(f"Error extracting text from date headers: {e}")
                                return f"Error extracting text from date headers: {str(e)}"
                            
                            # Remove line breaks and additional whitespace
                            try:
                                header_dates = [date.lower().split('\n')[0] for date in header_dates]
                            except Exception as e:
                                self.logger.error(f"Error processing date headers: {e}")
                                return f"Error processing date headers: {str(e)}"
                                
                            # Check if requested date is in the list of header dates
                            is_date_present = date.lower() in header_dates
                            
                            if not is_date_present:
                                try:
                                    next_button = self.self.driver.find_element(By.ID, "s-lc-whw-next-0")
                                    next_button.click()
                                    time.sleep(0.2)  # Allow page to load
                                except Exception as e:
                                    self.logger.error(f"Error clicking next button: {e}")
                                    return f"Error clicking next button: {str(e)}"
                            
                            iterations += 1
                        
                        # Optional: self.logger.info debug information
                        self.logger.info(f"Available Dates: {header_dates}")
                        self.logger.info(f"Requested Date: {date}")
                        self.logger.info(f"Date Present: {is_date_present}")
                        
                    
                        self.logger.info("\nhello")
                        mapped_library_names = library_map.get(str(library_name))
                        self.logger.info(f"Mapped library names: {mapped_library_names}")
                        
                        # Find library row
                        try:
                            library_row = self.self.driver.find_element(
                                By.XPATH, f"//tr[contains(., '{mapped_library_names}')]"
                            )
                        except Exception as e:
                            self.logger.error(f"Error finding library row: {e}")
                            return f"Error finding library row: {str(e)}"
                        
                        
                        self.logger.info("\nFound library row")

                        # Find date column index
                        try:
                            date_headers = self.self.driver.find_elements(By.XPATH, "//thead/tr/th/span[@class='s-lc-whw-head-date']")
                        except Exception as e:
                            self.logger.error(f"Error finding date headers: {e}")
                            return f"Error finding date headers: {str(e)}"
                        
                        self.logger.info(f"Found date_headers")
                        
                        date_column_index = None
                        for index, header in enumerate(date_headers, start=0):
                            self.logger.info(f"header.text.lower() = {header.text.lower()}")  
                            self.logger.info(f"date.lower() = {date.lower()}")  
                            if date.lower() == header.text.lower():
                                date_column_index = index+1 if index==0 else index
                                self.logger.info("\nFound date column index")
                                break

                        if date_column_index is None:
                            self.logger.info("\nNo date info found")
                            continue  # Skip if date not found
                        
                        self.logger.info(f"Found date column index {date_column_index}")
                        # Extract status
                        try:
                            status_cell = library_row.find_elements(By.TAG_NAME, "td")[date_column_index]
                            self.logger.info(f"Found library row elements : {status_cell}")
                        except Exception as e:
                            self.logger.error(f"Error finding status cell: {e}")
                            return f"Error finding status cell: {str(e)}"
                        try:
                            status = status_cell.find_element(By.CSS_SELECTOR, "span").text
                            self.logger.info(f"Found library status elements : {status}")
                        except Exception as e:
                            self.logger.info(f"Status cell HTML: {status_cell.get_attribute('outerHTML')}")
                            self.logger.error(f"Error extracting library status: {e}")
                            raise
                            break

                        # Append to results
                        library_result = {
                            'library': mapped_library_names,
                            'date': date,
                            'status': status
                        }
                        self.logger.info(f"mapping {library_result}")
                        results.append(library_result)

                # Convert results to formatted string for text_content
                self.logger.info(f"Results : {results}")
                for library in results:
                    lib_string = f"Library: {library['library']}\n"
                    lib_string += f"Date: {library['date']}\n"
                    lib_string += f"Status: {library['status']}\n"
                    
                    self.text_content.append({
                        'content': lib_string,
                        'metadata': {
                            'url': url,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }
                    })
                self.logger.info(self.text_content)
                
                return True

            except Exception as e:
                return f"Error retrieving library status: {str(e)}"
        
        elif 'asu.libcal.com' in url:
            # Navigate to the URL
            self.driver.get(url)
            
            # Wait for page to load
            
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'panel'))
                )
                
                # Parse page source with BeautifulSoup
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                
                
                # Reset text_content for new scrape
                self.text_content = []
                
                # Find all study room panels
                study_rooms = soup.find_all('div', class_='panel panel-default')
                
                for room in study_rooms:
                    # Extract study room name
                    study_room_name = room.find('h2', class_='panel-title').text.split('\n')[0].strip()
                    # Extract the date (consistent across all rooms)
                    date = room.find('p').text.strip()  # "Friday, December 6, 2024"
                    
                    # Find all available time slots
                    available_times = []
                    time_slots = room.find_all('div', class_='checkbox')
                    
                    for slot in time_slots:
                        time_text = slot.find('label').text.strip()
                        available_times.append(time_text)
                    
                    # Append to text_content
                    self.text_content.append({
                        'content': f"""Library: {optional_query}\nStudy Room: {study_room_name}\nDate: {date}\nAvailable slots: {', '.join(available_times)}""",
                        'metadata': {
                            'url': url,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'source_type': 'asu_web_scrape',
                        }
                    })
            except Exception as e:
                self.logger.error(f"Error extracting study room data: {e}")
                return "No Study Rooms Open Today"
            return True
        
        elif 'asu-shuttles.rider.peaktransit.com' in url:
            query = optional_query
            # Navigate to the URL
            try:
                # Navigate to the URL
                self.self.driver.get(url)
                time.sleep(3)
                # Wait for route list to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, '#route-list .route-block .route-name'))
                )
                # Target the route list container first
                route_list = self.self.driver.find_element(By.CSS_SELECTOR, "div#route-list.route-block-container")
                
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.route-block'))
                )
                route_blocks = route_list.find_elements(By.CSS_SELECTOR, "div.route-block")
                
                iterate_Y = 0
                results=[]
                button_times = 0
                iterate_X=0
                route = None
                for route_block in route_blocks:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.route-name'))
                    )
                    route_name = route_block.find_element(By.CSS_SELECTOR, "div.route-name")
                    self.logger.info("\nloacted routenames")
                    if "Tempe-Downtown" in route_name.text and  "Tempe-Downtown" in query:
                        button_times =5
                        route = route_name.text
                        route_block.click()
                        self.logger.info("\nclicked")
                        break
                    elif "Tempe-West" in route_name.text and "Tempe-West" in query:
                        button_times=5
                        route = route_name.text
                        route_block.click()
                        self.logger.info("\nclicked")
                        break
                    elif "Mercado" in route_name.text and "Mercado" in query:
                        button_times = 2
                        iterate_X = 12
                        iterate_Y = 8
                        route = route_name.text

                        route_block.click()
                        self.logger.info("\nMercado")
                        break
                    elif "Polytechnic" in route_name.text and "Polytechnic" in query:
                        button_times = 2
                        route_block.click()
                        iterate_X = 10
                        iterate_Y = 17
                        route = route_name.text

                    
                        self.logger.info("\nPolytechnic")
                        break
                
                time.sleep(2)
                
                # WebDriverWait(self.driver, 10).until(
                #     EC.presence_of_all_elements_located((By.XPATH, "//div[contains(@style, 'z-index: 106')]"))
                # )
                
                # try:
                #     zoom_out_button = WebDriverWait(self.driver, 10).until(
                #         EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Zoom out']"))
                #     )
                    
                #     for _ in range(button_times):
                #         zoom_out_button.click()
                #         time.sleep(0.5)  # Short pause between clicks
                
                # except Exception as e:
                #     self.logger.info(f"Error clicking zoom out button: {e}")
                
                # Replace the existing zoom control code with:

                try:
                    # First click on map camera controls button
                    camera_controls = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Map camera controls']"))
                    )
                    camera_controls.click()
                    time.sleep(0.5)  # Short pause to let controls expand

                    # Then find and click the zoom out button
                    zoom_out_button = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Zoom out']"))
                    )
                    
                    # Click zoom out button multiple times based on button_times parameter
                    for _ in range(button_times):
                        zoom_out_button.click()
                        time.sleep(0.5)  # Short pause between clicks

                except Exception as e:
                    self.logger.info(f"Error with map controls: {e}")
                    # Try alternative method using JavaScript
                    try:
                        zoom_script = """
                        var controls = document.querySelector('button[aria-label="Map camera controls"]');
                        if(controls) controls.click();
                        var zoomOut = document.querySelector('button[aria-label="Zoom out"]');
                        if(zoomOut) {
                            for(var i = 0; i < arguments[0]; i++) {
                                zoomOut.click();
                                await new Promise(r => setTimeout(r, 500));
                            }
                        }
                        """
                        self.self.driver.execute_script(zoom_script, button_times)
                    except Exception as e:
                        self.logger.info(f"Failed to zoom using JavaScript: {e}")

                map_div = None
                try:
                    # Method 1: JavaScript click
                    map_div = self.self.driver.find_element(By.CSS_SELECTOR, "div[aria-label='Map']")
                    self.self.driver.execute_script("arguments[0].click();", map_div)
                    self.logger.info("\nfirst method worked")
                
                except Exception as first_error:
                    try:
                        # Method 2: ActionChains click
                        map_div = self.driver.find_element(By.CSS_SELECTOR, "div[aria-label='Map']")
                        actions = ActionChains(self.driver)
                        actions.move_to_element(map_div).click().perform()
                        self.logger.info("\nsecond method worked")
                    
                    except Exception as second_error:
                        try:
                            # Method 3: Move and click with offset
                            map_div = self.driver.find_element(By.CSS_SELECTOR, "div[aria-label='Map']")
                            actions = ActionChains(self.driver)
                            actions.move_to_element_with_offset(map_div, 10, 10).click().perform()
                            self.logger.info("\nthird method worked")
                                 
                        except Exception as third_error:
                            self.logger.info(f"All click methods failed: {first_error}, {second_error}, {third_error}")

                
                
                actions = ActionChains(self.driver)
                if "Mercado" in query:
                    # Move map to different directions
                    directions_x = [
                        (300, 0), 
                    ]
                    directions_y = [
                        (0, 300),   
                    ]
                    
                    for i in range(0, iterate_X):
                        
                        for dx, dy in directions_x:
                            # Click and hold on map
                            actions.move_to_element(map_div).click_and_hold()
                            
                            # Move by offset
                            actions.move_by_offset(dx, dy)
                            
                            # Release mouse button
                            actions.release()
                            
                            # Perform the action
                            actions.perform()
                            self.logger.info("\nmoved")
                            # Wait a moment between movements
                    self.logger.info("\niterating over y")        
                    for i in range(0, iterate_Y):
                        for dx, dy in directions_y:
                            actions.move_to_element(map_div).click_and_hold()
                            actions.move_by_offset(dx, dy)
                            actions.release()
                            actions.perform()
                            self.logger.info("\nmoved")
                if "Polytechnic" in query:
                    self.logger.info("\npoly")
                    # Move map to different directions
                    directions_x = [
                        (-300, 0),
                    ]
                    directions_y = [
                        (0, -300),   
                    ]
                    
                    for i in range(0, iterate_X):
                        
                        for dx, dy in directions_x:
                            # Click and hold on map
                            actions.move_to_element(map_div).click_and_hold()
                            
                            # Move by offset
                            actions.move_by_offset(dx, dy)
                            actions.move_by_offset(dx, dy)
                            
                            # Release mouse button
                            actions.release()
                            
                            # Perform the action
                            actions.perform()
                            self.logger.info("\nmoved")
                            # Wait a moment between movements
                    self.logger.info("\niterating over y")        
                    for i in range(0, iterate_Y):
                        
                        for dx, dy in directions_y:
                            actions.move_to_element(map_div).click_and_hold()
                            actions.move_by_offset(dx, dy)
                            actions.release()
                            actions.perform()
                            self.logger.info("\nmoved")
                  
                map_markers = self.self.driver.find_elements(By.CSS_SELECTOR, 
                    'div[role="button"]  img[src="https://maps.gstatic.com/mapfiles/transparent.png"]')
                
                for marker in map_markers:
                    try:
                        parent_div = marker.find_element(By.XPATH, '..')
                        self.self.driver.execute_script("arguments[0].click();", parent_div)
                        
                        dialog = WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, 'div[role="dialog"]'))
                        )
                        
                        dialog_html = dialog.get_attribute('outerHTML')
                        soup = BeautifulSoup(dialog_html, 'html.parser')
                        
                        stop_name_elem = soup.find('div', class_='stop-name')
                        if stop_name_elem:
                            stop_name = stop_name_elem.find('h2').get_text(strip=True)
                            routes = soup.find_all('div', class_='route-name')
                            
                            station_routes = []
                            for route in routes:
                                route_name = route.get_text(strip=True)
                                bus_blocks = route.find_next_siblings('div', class_='bus-block')
                                
                                # Safer extraction of bus times
                                try:
                                    next_bus_time = bus_blocks[0].find('div', class_='bus-time').get_text(strip=True) if bus_blocks else 'N/A'
                                    second_bus_time = bus_blocks[1].find('div', class_='bus-time').get_text(strip=True) if len(bus_blocks) > 1 else 'N/A'
                                    
                                    station_routes.append({
                                        'Route': route_name,
                                        'Next Bus': next_bus_time,
                                        'Second Bus': second_bus_time
                                    })
                                except IndexError:
                                    # Skip routes without bus times
                                    continue
                            
                            # Only append if station_routes is not empty
                            if station_routes:
                                parsed_stations = [{
                                    'Station': stop_name,
                                    'Routes': station_routes
                                }]
                                results.extend(parsed_stations)
                        
                        
                        
                    
                    
                    except Exception as e:
                        # Log the error without stopping the entire process
                        self.logger.info(f"Error processing marker: {e}")
                        continue
                content = [
                            f"Station : {result['Station']}\n"
                            f"Route : {route['Route']}\n"
                            f"Next Bus : {route['Next Bus']}\n"
                            f"Second Bus : {route['Second Bus']}"
                            for result in results
                            for route in result['Routes']
                            if 'mins.' in route['Next Bus'] and 'mins.' in route['Second Bus']
                        ]
                content = set(content)  
                for c in content:
                    self.text_content.append({
                        'content': c,
                        'metadata': {
                            'url': url,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

                        }
                    })        
                return True
            
            except Exception as e:
                self.logger.info(f"Error extracting shuttle status: {e}")
                return False
            
        else:
            self.logger.error("NO CHOICE FOR SCRAPER!")
            
        return False
    
    
    async def engine_search(self, search_url: str =None, optional_query : str = None ) -> List[Dict[str, str]]:
        """Handle both Google search results and ASU Campus Labs pages using Selenium"""
        
        try:
            search_results = []
            await self.utils.update_text(f"Searching for [{urlparse(search_url).netloc}]({search_url})")
            # await self.discord_search(query=optional_query, channel_ids=[1323386884554231919,1298772258491203676,1256079393009438770,1256128945318002708], limit=30)
            # disabled temprarily
            wait = WebDriverWait(self.driver, 10)
            if (search_url):
                if 'google.com/search' in search_url:
                    self.driver.get(search_url)
                    # Wait for search results to load
                    self.logger.info(f"Searching for google links {search_url}")
                    try:
                        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.g')))
                    except Exception as e:
                        self.logger.error(f"Error waiting for search results: {e}")
                        
                    # Find all search result elements
                    try:
                        
                        results = self.driver.find_elements(By.CSS_SELECTOR, "div.tF2Cxc")[:3]
                        for index, result in enumerate(results, start=1):
                            # title = result.find_element(By.TAG_NAME, "h3").text
                            url = result.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                            search_results.append(url)
                            
                        
                        self.logger.info(f"Found {len(search_results)} : {search_results} Google search results")
                        
                        for url in search_results:
                            await self.scrape_content(url=url)

                    except Exception as e:
                        self.logger.error(f"Error extracting search results: {e}")
                                    # Handle ASU Campus Labs pages
                
                if 'asu.campuslabs.com/engage' in search_url:
                    self.driver.get(search_url)
                    if 'events' in search_url:
                        # Wait for events to load
                        events = wait.until(EC.presence_of_all_elements_located(
                            (By.CSS_SELECTOR, 'a[href*="/engage/event/"]')
                        ))
                        search_results = [
                            event.get_attribute('href') 
                            for event in events[:3]
                        ]
                        
                    elif 'organizations' in search_url:
                        # Wait for organizations to load
                        orgs = wait.until(EC.presence_of_all_elements_located(
                            (By.CSS_SELECTOR, 'a[href*="/engage/organization/"]')
                        ))
                        search_results = [
                            org.get_attribute('href') 
                            for org in orgs[:3]
                        ]
                        
                    elif 'news' in search_url:
                        # Wait for news items to load
                        news = wait.until(EC.presence_of_all_elements_located(
                            (By.CSS_SELECTOR, 'a[href*="/engage/news/"]')
                        ))
                        search_results = [
                            article.get_attribute('href') 
                            for article in news[:3]
                        ]
                        
                    self.logger.info(f"Found {len(search_results)} ASU Campus Labs results")
                    
                    for url in search_results:
                        await self.scrape_content(url=url)
                                 
                if 'x.com' in search_url or 'facebook.com' in search_url or "instagram.com" in search_url:
                    if optional_query:
                        self.logger.info("\nOptional query :: %s" % optional_query)
                        domain = urlparse(search_url).netloc
                        path = urlparse(search_url).path.strip("/")
                        query_part = f"{urllib.parse.quote(optional_query)}+{path}" if path else urllib.parse.quote(optional_query)
                        google_search_url = f"https://www.google.com/search?q={query_part}+site:{domain}"
                        self.logger.info("Google search url formed : {}".format(google_search_url))
                        await self.engine_search(search_url=google_search_url)                            
                
                if 'https://goglobal.asu.edu/scholarship-search' in search_url or 'https://onsa.asu.edu/scholarships'in search_url:
                    try:
                        # Get base domain based on URL
                        base_url = "https://goglobal.asu.edu" if "goglobal" in search_url else "https://onsa.asu.edu"
                        
                        self.driver.get(search_url)
                        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
                        
                        # Handle cookie consent for goglobal
                        try:
                            cookie_button = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, '.accept-btn'))
                            )
                            self.driver.execute_script("arguments[0].click();", cookie_button)
                            time.sleep(2)
                        except Exception as cookie_error:
                            self.logger.warning(f"Cookie consent handling failed: {cookie_error}")
                        
                        if optional_query:
                            self.logger.info("\nOptional query :: %s" % optional_query)
                            
                            # Parse query parameters
                            query_params = dict(param.split('=') for param in optional_query.split('&') if '=' in param)
                            
                            # Define filter mappings based on site
                            filter_mapping = {
                                'goglobal.asu.edu': {
                                    'academiclevel': '#edit-field-ss-student-type-target-id',
                                    'citizenship_status': '#edit-field-ss-citizenship-status-target-id',
                                    'gpa': '#edit-field-ss-my-gpa-target-id',
                                    # 'college': '#edit-field-college-ss-target-id',
                                },
                                'onsa.asu.edu': {
                                    'search_bar_query': 'input[name="combine"]',
                                    'citizenship_status': 'select[name="field_citizenship_status"]',
                                    'eligible_applicants': 'select[name="field_eligible_applicants"]',
                                    'focus': 'select[name="field_focus"]',
                                }
                            }
                            
                            # Determine which site's filter mapping to use
                            site_filters = filter_mapping['goglobal.asu.edu'] if 'goglobal.asu.edu' in search_url else filter_mapping['onsa.asu.edu']
                            
                            # Apply filters with robust error handling
                            for param, value in query_params.items():
                                if param in site_filters and value:
                                    try:
                                        filter_element = WebDriverWait(self.driver, 10).until(
                                            EC.element_to_be_clickable((By.CSS_SELECTOR, site_filters[param]))
                                        )
                                        
                                        # Scroll element into view
                                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", filter_element)
                                        time.sleep(1)
                                        
                                        # Handle different input types
                                        if filter_element.tag_name == 'select':
                                            Select(filter_element).select_by_visible_text(value)
                                        elif filter_element.tag_name == 'input':
                                            filter_element.clear()
                                            filter_element.send_keys(value)
                                            filter_element.send_keys(Keys.ENTER)
                                        
                                        time.sleep(1)
                                    except Exception as filter_error:
                                        self.logger.warning(f"Could not apply filter {param}: {filter_error}")
                            
                            # Click search button with multiple retry mechanism
                            search_button_selectors = ['input[type="submit"]', 'button[type="submit"]', '.search-button']
                            for selector in search_button_selectors:
                                try:
                                    search_button = WebDriverWait(self.driver, 10).until(
                                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                                    )
                                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", search_button)
                                    time.sleep(1)
                                    self.driver.execute_script("arguments[0].click();", search_button)
                                    break
                                except Exception as e:
                                    self.logger.warning(f"Search button click failed for selector {selector}: {e}")
                        
                        # Extract scholarship links with improved URL construction
                        link_selectors = {
                            'goglobal': 'td[headers="view-title-table-column"] a',
                            'onsa': 'td a'
                        }
                        
                        current_selector = link_selectors['goglobal'] if "goglobal" in search_url else link_selectors['onsa']
                        
                        scholarship_links = WebDriverWait(self.driver, 10).until(
                            EC.presence_of_all_elements_located((By.CSS_SELECTOR, current_selector))
                        )
                        
                        for link in scholarship_links[:3]:
                            href = link.get_attribute('href')
                            if href:
                                if href.startswith('/'):
                                    search_results.append(f"{base_url}{href}")
                                elif href.startswith('http'):
                                    search_results.append(href)
                                else:
                                    search_results.append(f"{base_url}/{href}")
                        
                        self.logger.info(f"Found {len(search_results)} scholarship links - ")
                        self.logger.info(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                        self.logger.info(f"{search_results}")
                        self.logger.info(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                        
                        for url in search_results:
                            await self.scrape_content(url=url)
                        
                    except Exception as e:
                        self.logger.error(f"Error in scholarship search: {str(e)}")

                
                if 'catalog.apps.asu.edu' in search_url or  'search.lib.asu.edu' in search_url :
                    await self.scrape_content(search_url, selenium=True)
                
                if 'https://app.joinhandshake.com/stu/postings' in search_url or 'lib.asu.edu' in search_url or "asu.libcal.com" in search_url or "asu-shuttles.rider.peaktransit.com" in search_url:                    
                    await self.scrape_content(search_url, selenium=True, optional_query=optional_query)
                # finally:
                #     self.driver.quit()
 
            return self.text_content
                
        except Exception as e:
            self.logger.error(f"Error in search: {str(e)}")
            return []
        
    # async def discord_search(self, query: str, channel_ids: List[int], limit: int = 40) -> List[Dict[str, str]]:
    #     if not self.discord_client:
    #         self.logger.info(f"Could not initialize discord_client {self.discord_client}")
    #         return []
        
    #     messages = []
    #     await self.utils.update_text("Searching the Sparky Discord Server")
        
    #     for channel_id in channel_ids:
    #         channel = self.discord_client.get_channel(channel_id)
            
    #         if not channel:
    #             self.logger.info(f"Could not access channel with ID {channel_id}")
    #             continue
            
    #         if isinstance(channel, discord.TextChannel):
    #             async for message in channel.history(limit=limit):
    #                 messages.append(self._format_message(message))
    #         elif isinstance(channel, discord.ForumChannel):
    #             async for thread in channel.archived_threads(limit=limit):
    #                 async for message in thread.history(limit=limit):
    #                     messages.append(self._format_message(message))
            
    #         if len(messages) >= limit:
    #             break
            
    #     print(messages)
        
    #     for message in messages[:limit]:
    #         self.text_content.append({
    #             'content': message['content'],
    #             'metadata': {
    #                 'url': message['url'],
    #                 'timestamp': message['timestamp'],
    #             }
    #         })

        
    #     return True
    
    # def _format_message(self, message: discord.Message) -> Dict[str, str]:
    #     timestamp = message.created_at.strftime("%Y-%m-%d %H:%M:%S")
        
    #     formatted_content = (
    #         f"Sent by: {message.author.name} {timestamp}\n"
    #         f"Message content: {message.content}"
    #     )
        
    #     return {
    #         'url': message.jump_url,
    #         'content': formatted_content,
    #         'timestamp': timestamp
    #     }
    
    #  else:
    #     if 'x.com' in search_url or 'twitter.com' in search_url:
    #         try:
    #             try:
    #                 WebDriverWait(self.driver, 30).until(
    #                     EC.presence_of_all_elements_located((By.TAG_NAME, 'body'))
    #                 )
    #             except Exception as e:
    #                 self.logger.warning(f"Timeout waiting for tweets to load {str(e)}")
                
    #             self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    #             time.sleep(3)
                
    #             # Define tweet selectors
    #             tweet_selectors = [
    #                 'article[data-testid="tweet"]',
    #                 'div[data-testid="cellInnerDiv"]',
    #                 'div[role="article"]',
                    
    #             ]
                
    #             # Find tweet articles
    #             try:
                        
    #                 tweet_articles = []
    #                 for selector in tweet_selectors:
    #                     tweet_articles = self.driver.find_elements(By.CSS_SELECTOR, selector)
    #                     if tweet_articles:
    #                         break
    #             except Exception as e:
    #                 self.logger.error(f"Error finding tweet articles: {str(e)}")
                    
                    
    #             if not tweet_articles:
    #                 self.logger.error('No tweet articles found')
    #                 return []
                
    #             # Extract top 3 tweet links
    #             tweet_links = []
    #             for article in tweet_articles[:3]:
    #                 try:
    #                     link_selectors = [
    #                         'a[href*="/status/"]',
    #                         'a[dir="ltr"][href*="/status/"]'
    #                     ]
                        
    #                     for selector in link_selectors:
    #                         try:
    #                             timestamp_link = article.find_element(By.CSS_SELECTOR, selector)
    #                             tweet_url = timestamp_link.get_attribute('href')
    #                             if tweet_url:
    #                                 tweet_links.append(tweet_url)
    #                                 break
    #                         except:
    #                             continue
    #                 except Exception as inner_e:
    #                     self.logger.error(f"Error extracting individual tweet link: {str(inner_e)}")
                
    #             self.logger.info(tweet_links)
    #             search_results.extend(tweet_links)
    #             self.logger.info(f"Found {len(tweet_links)} X (Twitter) links")
                
    #         except Exception as e:
    #             self.logger.error(f"X.com tweet link extraction error: {str(e)}")

        
    #     elif 'instagram.com' in search_url:
    #         try:
    #             instagram_post_selectors = [
    #                 'article[role="presentation"]',
    #                 'div[role="presentation"]',
    #                 'div[class*="v1Nh3"]'
    #             ]
                
    #             instagram_link_selectors = [
    #                 'a[href*="/p/"]',
    #                 'a[role="link"][href*="/p/"]'
    #             ]
                
    #             instagram_articles = []
    #             for selector in instagram_post_selectors:
    #                 instagram_articles = self.driver.find_elements(By.CSS_SELECTOR, selector)
    #                 if instagram_articles:
    #                     break
                
    #             instagram_links = []
    #             for article in instagram_articles[:3]:
    #                 for link_selector in instagram_link_selectors:
    #                     try:
    #                         post_link = article.find_element(By.CSS_SELECTOR, link_selector)
    #                         insta_url = post_link.get_attribute('href')
    #                         if insta_url and insta_url not in instagram_links:
    #                             instagram_links.append(insta_url)
    #                             break
    #                     except Exception as insta_link_error:
    #                         continue
                
    #             search_results.extend(instagram_links)
    #             self.logger.info(f"Found {len(instagram_links)} Instagram post links")
            
    #         except Exception as instagram_error:
    #             self.logger.error(f"Instagram link extraction error: {str(instagram_error)}")
