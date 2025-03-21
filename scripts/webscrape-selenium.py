from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time

# Set up Chrome options to add headers and user-agent
chrome_options = Options()
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # Prevent detection as a bot

# Initialize the WebDriver with options
driver = webdriver.Chrome(options=chrome_options)

try:
    # Open Google
    driver.get("https://www.google.com/search?q=asu+social+media&oq=asu+social+media")

    # # Find the search bar and input the query
    # search_box = driver.find_element(By.NAME, "q")
    # query = "Selenium Python tutorial"
    # search_box.send_keys(query)
    # search_box.send_keys(Keys.RETURN)

    # Wait for results to load
    time.sleep(2)

    # Get the top three search results
    results = driver.find_elements(By.CSS_SELECTOR, "div.tF2Cxc")[:3]

    # Log the title and URL of each result
    for index, result in enumerate(results, start=1):
        title = result.find_element(By.TAG_NAME, "h3").text
        url = result.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
        print(f"Result {index}:")
        print(f"Title: {title}")
        print(f"URL: {url}")
        print()

finally:
    # Close the WebDriver
    driver.quit()