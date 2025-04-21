#!/usr/bin/env python3
from selenium import webdriver, common
from selenium.webdriver.common.by   import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui  import WebDriverWait
from selenium.webdriver.support     import expected_conditions as EC
import time

URL = "https://www.asu.edu/map/interactive/"

driver = webdriver.Chrome()
driver.maximize_window()
driver.get(URL)
wait = WebDriverWait(driver, 30)

# –– iframe
driver.switch_to.frame(wait.until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "iframe.map"))
))

# –– search
box = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input.searchInput")))
box.clear();  box.send_keys("LSE", Keys.ENTER)

# –– popup header
title_el = wait.until(EC.visibility_of_element_located((
    By.CSS_SELECTOR, "div.esri-popup__title, div.header"
)))
print("Building :", title_el.text.strip())

# –– scrollable pane inside popup
pane = driver.find_element(By.CSS_SELECTOR, "div.contentPane")

# find the additional‑info anchor (will be present but out of view initially)
add_link = pane.find_element(
    By.XPATH, ".//a[contains(., 'Additional Building Information')]"
)

# scroll it into view, then click via JS to avoid overlay issues
driver.execute_script("arguments[0].scrollIntoView({block:'center'});", add_link)
driver.execute_script("arguments[0].click();", add_link)

# –– new tab → switch
time.sleep(1)
driver.switch_to.window(driver.window_handles[-1])

# –– grab first Google‑Maps link on that page
gmaps = wait.until(EC.presence_of_element_located((
    By.XPATH, "//a[contains(@href,'google.com/maps')]"
))).get_attribute("href")

print("Google‑Maps URL :", gmaps)

time.sleep(5)
driver.quit()
