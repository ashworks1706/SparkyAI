import time
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# For explicit waits
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_asu_workday_jobs(keyword="arts"):
    """
    1) Opens the ASU Workday student jobs page
    2) Waits 2 min for manual login
    3) Refreshes page
    4) Locates the search bar (data-automation-id='textInputBox')
    5) Types in 'keyword' and presses Enter
    6) Scrapes only the top 5 jobs in the results
    7) Saves everything to a CSV
    """

    service = Service("/usr/local/bin/chromedriver")  # Adjust if needed
    options = webdriver.ChromeOptions()

    # Optional: reuse your Chrome profile
    # options.add_argument("user-data-dir=/Users/<USERNAME>/Library/Application Support/Google/Chrome/Default")

    driver = webdriver.Chrome(service=service, options=options)
    url = "https://www.myworkday.com/asu/d/task/1422$3898.htmld"
    driver.get(url)

    print("Please log in if prompted. Waiting 2 minutes…")
    time.sleep(50)  # manual login

    # Extra wait in case of post-login re-render
    time.sleep(5)
    driver.refresh()
    time.sleep(3)

    SEARCH_BAR_SELECTOR = "input[data-automation-id='textInputBox']"

    try:
        # Wait for the search box to appear in the DOM
        search_box = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, SEARCH_BAR_SELECTOR))
        )

        # Possibly scroll into view
        driver.execute_script("arguments[0].scrollIntoView(true);", search_box)
        time.sleep(1)

        # Wait until it's clickable
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, SEARCH_BAR_SELECTOR))
        )

        # Try normal click
        try:
            search_box.click()
        except:
            # If normal click fails, do a JS click
            driver.execute_script("arguments[0].click();", search_box)

        # Clear if needed, then send keys
        search_box.clear()
        search_box.send_keys(keyword)
        search_box.send_keys(Keys.RETURN)
        print(f"Searched for '{keyword}'. Waiting for results to load...")

    except Exception as e:
        print("Could not find or use the search box. Check the selector or page structure.\n", e)
        driver.quit()
        return

    time.sleep(5)  # wait for search results

    # Now gather the job results
    job_selector = "div[role='link'][data-automation-label]"
    job_divs = driver.find_elements(By.CSS_SELECTOR, job_selector)
    print(f"Found {len(job_divs)} job items on the page for keyword='{keyword}'.")

    # Only scrape the top 5
    top_job_divs = job_divs[:5]

    all_jobs_data = []

    for index, job_div in enumerate(top_job_divs, start=1):
        job_title = job_div.get_attribute("data-automation-label") or "N/A"
        print(f"\nOpening job #{index}: {job_title}")

        # Open detail in a new tab
        action_key = Keys.COMMAND  # or Keys.CONTROL (Windows)
        webdriver.ActionChains(driver).key_down(action_key).click(job_div).key_up(action_key).perform()

        time.sleep(2)
        driver.switch_to.window(driver.window_handles[-1])

        # Wait for detail panel
        try:
            detail_panel = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-automation-id='jobPosting']"))
            )
        except:
            print("WARNING: Timed out waiting for job detail overlay.")
            detail_panel = None

        job_data = {"title": job_title}

        # Attempt to read <h1> if it exists
        try:
            h1_elem = driver.find_element(By.CSS_SELECTOR, "h1")
            job_data["detail_header"] = h1_elem.text.strip()
        except:
            job_data["detail_header"] = "N/A"

        # Grab detail text
        if detail_panel:
            job_data["detail_text"] = detail_panel.text.strip()
        else:
            job_data["detail_text"] = driver.find_element(By.TAG_NAME, "body").text.strip()

        all_jobs_data.append(job_data)
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        time.sleep(2)

    # Write results to CSV
    csv_filename = f"asu_workday_jobs_{keyword}.csv"
    fieldnames = ["title", "detail_header", "detail_text"]
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_jobs_data)

    print(f"\nDone! Saved {len(all_jobs_data)} job records to {csv_filename}")
    driver.quit()

if __name__ == "__main__":
    # Prompt the user for a keyword, then only scrape the top 5 results
    user_keyword = input("Enter a keyword to search: ")
    if not user_keyword.strip():
        user_keyword = "arts"  # fallback default
    scrape_asu_workday_jobs(keyword=user_keyword)
