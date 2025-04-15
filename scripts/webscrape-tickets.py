from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time

# Setting variables that can be useful for queries
ticketed_sports_dic = ["football", "men's basketball", "women's basketball", "hockey", "baseball", "gymnastics", "volleyball", "softball", "wrestling"]
non_ticketed_sports_dic = ["cross country", "golf", "swimming and diving", "tennis", "track and field"]
sport = ""
list_of_games = []

# get either ticketed or non ticketed sport, or assuming user wants any sport
response = input("Enter sport: ")
if response in ticketed_sports_dic:
    sport = response
elif response in non_ticketed_sports_dic:
    print("Sport is not ticketed")
    sport = response
else:
    sport = ""

# Open chrome
options = Options()
options.add_argument("--headless=new") 
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920,1080')
options.add_argument('--ignore-certificate-errors')
options.add_argument('--disable-extensions')
options.add_argument('--no-first-run')
options.add_argument("--disable-blink-features=AutomationControlled") 

driver = webdriver.Chrome(options=options)
driver.get("https://sundevils.com/tickets")

# Find search bar
wait = WebDriverWait(driver, 10)
search_bar = wait.until(
    EC.presence_of_element_located((By.CLASS_NAME, "sc-gEkIjz"))
)

# Clear any existing text and type the search term
search_bar.clear()
search_bar.send_keys(sport)
search_bar.send_keys(Keys.RETURN)  # Press Enter
time.sleep(2)

# Find relevant games
games = driver.find_elements(By.CLASS_NAME, "sc-lizKOf")
for game in games:
    if game.text.strip() == "":
        continue

    game_information = game.text.split("\n")

    # data we want to send the user
    sport = game_information[0]
    date = game_information[1] + " " + game_information[2]
    time = game_information[3] + " " + game_information[4]
    location = game_information[5]
    rival_team = game_information[6]

    # game themes
    themes = []
    for theme in game_information[7:]:
        if theme != "Buy tickets" and theme != "Event details" and theme != "History":
            themes.append(theme)

    # game links
    links = game.find_elements(By.TAG_NAME, "a")
    extra_links = []
    event_link = None
    ticket_link = None
    for link in links:
        if link.text == "Buy tickets":
            ticket_link = link.get_attribute("href")
        elif link.text == "Event details":
            event_link = link.get_attribute("href")
        else:
            extra_links.append(link.get_attribute("href"))

    # object store
    game_details = {
        "sport": sport,
        "date": date,
        "time": time,
        "location": location,
        "rival_team": rival_team,
        "themes": themes,
        "ticket_link": ticket_link if ticket_link else None,
        "event_link": event_link if event_link else None,
        "extra_links": extra_links if extra_links else None,
    }
    #top 10 games
    list_of_games.append(game_details)

    print(game_details)
    print("----------------")

# Handle whenever query does not return a result
try:
    no_games = driver.find_element(By.CLASS_NAME, "sc-gdyeKB")
    if no_games.text == "No upcoming games. Check back soon.":
        print("No upcoming games. Check back soon.")
except:
    pass

# Close the browser
driver.quit()
