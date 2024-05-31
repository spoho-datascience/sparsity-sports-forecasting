### functions 

# OddsPortal scraper functions 


import os
import urllib
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from selenium.webdriver.common.keys import Keys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from selenium.webdriver.common.action_chains import ActionChains
import signal

#from create_clean_table import *

global DRIVER_LOCATION
DRIVER_LOCATION = "/opt/chromedriver"

global TYPE_ODDS
TYPE_ODDS = 'CLOSING' # you can change to 'OPENING' if you want to collect opening odds, any other value will make the program collect CLOSING odds

def get_opening_odd(xpath):
    # I. Get the raw data by hovering and collecting
    data = driver.find_element("xpath", xpath)
    hov = ActionChains(driver).move_to_element(data)
    hov.perform()
    data_in_the_bubble = driver.find_element("xpath", "//*[@id='tooltiptext']")
    hover_data = data_in_the_bubble.get_attribute("innerHTML")

    # II. Extract opening odds
    b = re.split('<br>', hover_data)
    c = [re.split('</strong>',y)[0] for y in b][-2]
    opening_odd = re.split('<strong>', c)[1]

    #print(opening_odd)
    return(opening_odd)
    
    
def fi(a):
    try:
        driver.find_element("xpath", a).text
    except:
        return False

def ffi(a):
    if fi(a) != False :
        return driver.find_element("xpath", a).text
def fffi(a):
    if TYPE_ODDS == 'OPENING':
        try:
            return get_opening_odd(a) 
        except:
            return ffi(a)  
    else:
        return(ffi(a))

def fi2(a):
    try:
        driver.find_element("xpath", a).click()
    except:
        return False

def ffi2(a):
    if fi2(a) != False :
        fi2(a)
        return(True)
    else:
        return(None)

def get_matches_links(driver, link):
    driver.get(link)
    reject_ads()
    links = driver.find_elements(by=By.CSS_SELECTOR, value="a[href*='/football/spain/laliga-2021-2022/']")
    action_links = [l.get_attribute('href') for l in links]
    match_action_links = []
    for action_link in action_links:
        if ('results' not in action_link) and ('standings' not in action_link) and (not action_link.endswith('laliga-2021-2022/')) and (action_link not in match_action_links):
            match_action_links.append(action_link)
    print(f"Found {len(match_action_links)} links to matches.")
    return match_action_links

def get_match_info(driver, match_link):
    driver.get(match_link)
    reject_ads()
    time.sleep(0.5)
    odds_row = driver.find_element(
        by=By.XPATH, value="//*[text()='Average']"
    ).find_element(
        by=By.XPATH, value="../.."
    ).find_elements(
        by=By.XPATH, value=".//div"
    )
    match_dict = {}
    for odds_name, odds_info in [('odds_home', odds_row[1]), ('odds_draw', odds_row[3]), ('odds_away', odds_row[5])]:
        match_dict[odds_name] = odds_info.text

    team_logo_images = driver.find_elements(
        by=By.CSS_SELECTOR, value="img[src*='/team-logo/']"
    )
    home_away = ['home', 'away']
    for team_i, team_logo_img in enumerate(team_logo_images):
        results_divs = team_logo_img.find_element(
            by=By.XPATH, value="../.."
        ).find_elements(
            by=By.XPATH, value=".//div"
        )
        team_name = results_divs[0].text
        score = int(results_divs[1].text)
        match_dict[f"{home_away[team_i]}_team"] = team_name
        match_dict[f"{home_away[team_i]}_score"] = score
    
    match_date = driver.find_element(
        by=By.CLASS_NAME, value="bg-event-start-time"
    ).find_elements(
        by=By.XPATH, value="..//div"
    )[1].text
    match_dict["match_date"] = match_date
    return match_dict

def get_data_next_games_typeC(i, link):
    driver.get(link)
    reject_ads()
    target = '//*[@id="tournamentTable"]/tbody/tr[{}]/td[2]/a[2]'.format(i)
    a = ffi2(target)

    L = []


    if a == True:
        print('We wait 4 seconds')
        time.sleep(4)
        # Now we collect all bookmaker
        for j in range(1,30): # only first 10 bookmakers displayed
            Book = ffi('//*[@id="odds-data-table"]/div[1]/table/tbody/tr[{}]/td[1]/div/a[2]'.format(j)) # first bookmaker name
            Odd_1 = fffi('//*[@id="odds-data-table"]/div[1]/table/tbody/tr[{}]/td[2]/a'.format(j)) # first home odd
            Odd_X = fffi('//*[@id="odds-data-table"]/div[1]/table/tbody/tr[{}]/td[3]/a'.format(j)) # draw odd
            Odd_2 = fffi('//*[@id="odds-data-table"]/div[1]/table/tbody/tr[{}]/td[4]/a'.format(j)) # first away odd
            match = ffi('//*[@id="col-content"]/h1') # match teams
            final_score = ffi('//*[@id="event-status"]')
            date = ffi('//*[@id="col-content"]/p[1]') # Date and time
            print(match, Book, Odd_1, Odd_X, Odd_2, date, final_score, i, '/ 30 ')
            L = L + [(match, Book, Odd_1, Odd_X, Odd_2, date, final_score)]
            
            Book = ffi('//*[@id="odds-data-table"]/div[1]/table/tbody/tr[{}]/td[1]/div/a[2]'.format(j)) # first bookmaker name
            Odd_1 = fffi('//*[@id="odds-data-table"]/div[1]/table/tbody/tr[{}]/td[2]/div'.format(j)) # first home odd
            Odd_X = fffi('//*[@id="odds-data-table"]/div[1]/table/tbody/tr[{}]/td[3]/div'.format(j)) # draw odd
            Odd_2 = fffi('//*[@id="odds-data-table"]/div[1]/table/tbody/tr[{}]/td[4]/div'.format(j)) # first away odd
            match = ffi('//*[@id="col-content"]/h1') # match teams
            final_score = ffi('//*[@id="event-status"]')
            date = ffi('//*[@id="col-content"]/p[1]') # Date and time
            print(match, Book, Odd_1, Odd_X, Odd_2, date, final_score, i, '/ 30 ')
            L = L + [(match, Book, Odd_1, Odd_X, Odd_2, date, final_score)]
            
    target = '//*[@id="tournamentTable"]/tbody/tr[{}]/td[2]/a'.format(i)
    a = ffi2(target)
    if a == True:
        print('We wait 4 seconds')
        time.sleep(4)
        # Now we collect all bookmaker
        for j in range(1,30): # only first 10 bookmakers displayed
            Book = ffi('//*[@id="odds-data-table"]/div[1]/table/tbody/tr[{}]/td[1]/div/a[2]'.format(j)) # first bookmaker name
            Odd_1 = fffi('//*[@id="odds-data-table"]/div[1]/table/tbody/tr[{}]/td[2]/a'.format(j)) # first home odd
            Odd_X = fffi('//*[@id="odds-data-table"]/div[1]/table/tbody/tr[{}]/td[3]/a'.format(j)) # draw odd
            Odd_2 = fffi('//*[@id="odds-data-table"]/div[1]/table/tbody/tr[{}]/td[4]/a'.format(j)) # first away odd
            match = ffi('//*[@id="col-content"]/h1') # match teams
            final_score = ffi('//*[@id="event-status"]')
            date = ffi('//*[@id="col-content"]/p[1]') # Date and time
            print(match, Book, Odd_1, Odd_X, Odd_2, date, final_score, i, '/ 30 ')
            L = L + [(match, Book, Odd_1, Odd_X, Odd_2, date, final_score)]
            
            Book = ffi('//*[@id="odds-data-table"]/div[1]/table/tbody/tr[{}]/td[1]/div/a[2]'.format(j)) # first bookmaker name
            Odd_1 = fffi('//*[@id="odds-data-table"]/div[1]/table/tbody/tr[{}]/td[2]/div'.format(j)) # first home odd
            Odd_X = fffi('//*[@id="odds-data-table"]/div[1]/table/tbody/tr[{}]/td[3]/div'.format(j)) # draw odd
            Odd_2 = fffi('//*[@id="odds-data-table"]/div[1]/table/tbody/tr[{}]/td[4]/div'.format(j)) # first away odd
            match = ffi('//*[@id="col-content"]/h1') # match teams
            final_score = ffi('//*[@id="event-status"]')
            date = ffi('//*[@id="col-content"]/p[1]') # Date and time
            print(match, Book, Odd_1, Odd_X, Odd_2, date, final_score, i, '/ 30 ')
            L = L + [(match, Book, Odd_1, Odd_X, Odd_2, date, final_score)]
            


    return(L)

def scrape_season_page(driver, page, sport, country, tournament, SEASON):
    link = 'https://www.oddsportal.com/{}/{}/{}-{}/results/#/page/{}'.format(sport,country,tournament,SEASON,page)
    DATA = []
    match_links = get_matches_links(driver, link)
    for match in match_links:
        print(match)
        DATA.append(get_match_info(driver, match))
    print(f"Extracted {len(DATA)} matches.")
    return DATA

def scrape_page_next_games_typeC(country,sport,  tournament, nmax = 20):
    link = 'https://www.oddsportal.com/{}/{}/{}/'.format(sport, country,tournament)
    DATA = []
    for i in range(1,nmax):
        print(i)
        content = get_data_next_games_typeC(i, link)
        if content != None:
            DATA = DATA + content
    print(DATA)
    return(DATA)
def scrape_page_next_games_typeC(country,sport,  tournament, nmax = 20):
    link = 'https://www.oddsportal.com/{}/{}/{}/'.format(sport, country,tournament)
    DATA = []
    for i in range(1,nmax):
        print(i)
        content = get_data_next_games_typeC(i, link)
        if content != None:
            DATA = DATA + content
    print(DATA)
    return(DATA)

def scrape_page_current_season_typeC(page,sport, country, tournament):
    link = 'https://www.oddsportal.com/{}/{}/{}/results/#/page/{}'.format(sport,country,tournament,page)
    DATA = []
    for i in range(1,100):
        content = get_matches_links(i, link)
        if content != None:
            DATA = DATA + content
    print(DATA)
    return(DATA)

def scrape_season_tournament(driver, sport, tournament, country, SEASON, max_page = 25):
    DATA_ALL = []
    for page in range(1, max_page + 1):
        print('We start to scrape the page n°{}'.format(page))
        data = scrape_season_page(driver, page, sport, country, tournament, SEASON)
        DATA_ALL = DATA_ALL + [y for y in data if y != None]
        driver.close()

    data_df = pd.DataFrame(DATA_ALL)

    try:
        data_df.columns = ['TeamsRaw', 'Bookmaker', 'OddHome','OddDraw', 'OddAway', 'DateRaw' ,'ScoreRaw']
    except:
        print('Function crashed, probable reason : no games scraped (empty season)')
        return(1)
    ##################### FINALLY WE CLEAN THE DATA AND SAVE IT ##########################
    '''Now we simply need to split team names, transform date, split score'''

    # (0) Filter out None rows
    data_df = data_df[~data_df['Bookmaker'].isnull()].dropna().reset_index()
    data_df["TO_KEEP"] = 1
    for i in range(len(data_df["TO_KEEP"])):
        if len(re.split(':',data_df["ScoreRaw"][i]))<2 :
            data_df["TO_KEEP"].iloc[i] = 0

    data_df = data_df[data_df["TO_KEEP"] == 1]

    # (a) Split team names
    data_df["Home_id"] = [re.split(' - ',y)[0] for y in data_df["TeamsRaw"]]
    data_df["Away_id"] = [re.split(' - ',y)[1] for y in data_df["TeamsRaw"]]
    # (b) Transform date
    data_df["Date"] = [re.split(', ',y)[1] for y in data_df["DateRaw"]]
    # (c) Split score
    data_df["Score_home"] = [re.split(':',y)[0][-2:] for y in data_df["ScoreRaw"]]
    data_df["Score_away"] = [re.split(':',y)[1][:2] for y in data_df["ScoreRaw"]]
    # (e) Set season column
    data_df["Season"] = SEASON
    # Finally we save results
    if not os.path.exists('./{}_FULL'.format(tournament)):
        os.makedirs('./{}_FULL'.format(tournament))
    if not os.path.exists('./{}'.format(tournament)):
        os.makedirs('./{}'.format(tournament))

    data_df.to_csv('./{}_FULL/{}_{}_FULL.csv'.format(tournament,tournament, SEASON), sep=';', encoding='utf-8', index=False)
    data_df[['Home_id', 'Away_id', 'Bookmaker', 'OddHome','OddDraw', 'OddAway', 'Date', 'Score_home', 'Score_away','Season']].to_csv('./{}/{}_{}.csv'.\
        format(tournament,tournament, SEASON), sep=';', encoding='utf-8', index=False)

    return(data_df)

def scrape_current_season_typeC(tournament, sport, country, SEASON, max_page = 25):
    global driver
    ############### NOW WE SEEK TO SCRAPE THE ODDS AND MATCH INFO################################
    DATA_ALL = []
    for page in range(1, max_page + 1):
        print('We start to scrape the page n°{}'.format(page))
        try:
            driver.quit() # close all widows
        except:
            pass

        
        data = scrape_page_current_season_typeC(page, sport, country, tournament)
        DATA_ALL = DATA_ALL + [y for y in data if y != None]
        driver.close()
    data_df = pd.DataFrame(DATA_ALL)
    try:
        data_df.columns = ['TeamsRaw', 'Bookmaker', 'OddHome','OddDraw', 'OddAway', 'DateRaw' ,'ScoreRaw']
    except:
        print('Function crashed, probable reason : no games scraped (empty season)')
        return(1)
    ##################### FINALLY WE CLEAN THE DATA AND SAVE IT ##########################
    '''Now we simply need to split team names, transform date, split score'''

    # (0) Filter out None rows
    data_df = data_df[~data_df['Bookmaker'].isnull()].dropna().reset_index()
    data_df["TO_KEEP"] = 1
    for i in range(len(data_df["TO_KEEP"])):
        if len(re.split(':',data_df["ScoreRaw"][i]))<2 :
            data_df["TO_KEEP"].iloc[i] = 0

    data_df = data_df[data_df["TO_KEEP"] == 1]
    # (a) Split team names
    data_df["Home_id"] = [re.split(' - ',y)[0] for y in data_df["TeamsRaw"]]
    data_df["Away_id"] = [re.split(' - ',y)[1] for y in data_df["TeamsRaw"]]
    # (b) Transform date
    data_df["Date"] = [re.split(', ',y)[1] for y in data_df["DateRaw"]]
    # (c) Split score
    data_df["Score_home"] = [re.split(':',y)[0][-2:] for y in data_df["ScoreRaw"]]
    data_df["Score_away"] = [re.split(':',y)[1][:2] for y in data_df["ScoreRaw"]]
    # (e) Set season column
    data_df["Season"] = SEASON
    # Finally we save results
    if not os.path.exists('./{}_FULL'.format(tournament)):
        os.makedirs('./{}_FULL'.format(tournament))
    if not os.path.exists('./{}'.format(tournament)):
        os.makedirs('./{}'.format(tournament))

    data_df.to_csv('./{}_FULL/{}_CurrentSeason_FULL.csv'.format(tournament,tournament), sep=';', encoding='utf-8', index=False)
    data_df[['Home_id', 'Away_id', 'Bookmaker', 'OddHome','OddDraw', 'OddAway', 'Date', 'Score_home', 'Score_away','Season']].\
        to_csv('./{}/{}_CurrentSeason.csv'.format(tournament,tournament), sep=';', encoding='utf-8', index=False)
    return(data_df)
    
def scrape_league(driver, Season, sport, country1, tournament1, nseason, max_page = 25):
    long_season = (len(Season) > 6) # indicates whether Season is in format '2010-2011' or '2011' depends on the league) 
    Season = int(Season[0:4])
    for i in range(nseason):
        SEASON1 = '{}'.format(Season)
        if long_season:
          SEASON1 = '{}-{}'.format(Season, Season+1)
        print('We start to collect season {}'.format(SEASON1))
        scrape_season_tournament(driver = driver, sport = sport, tournament = tournament1, country = country1, SEASON = SEASON1, max_page = max_page)
        print('We finished to collect season {} !'.format(SEASON1))
        Season+=1

    # Finally we merge all files
    file1 = pd.read_csv('./{}/'.format(tournament1) + os.listdir('./{}/'.format(tournament1))[0], sep=';')
    print(os.listdir('./{}/'.format(tournament1))[0])
    for filename in os.listdir('./{}/'.format(tournament1))[1:]:
        file = pd.read_csv('./{}/'.format(tournament1) + filename, sep=';')
        print(filename)
        file1 = file1.append(file)

    file1 = file1.reset_index()

    #Correct falsly collected data for away (in case of 1X2 instead of H/A odds)
    return(file1)
    

def scrape_next_games_typeC(tournament, sport, country, SEASON, nmax = 30):
    global driver
    ############### NOW WE SEEK TO SCRAPE THE ODDS AND MATCH INFO################################
    DATA_ALL = []
    try:
        driver.quit() # close all widows
    except:
        pass

 
    driver = webdriver.Chrome(executable_path = DRIVER_LOCATION)
    data = scrape_page_next_games_typeC(country, sport, tournament, nmax)
    DATA_ALL = DATA_ALL + [y for y in data if y != None]
    driver.close()

    data_df = pd.DataFrame(DATA_ALL)
  
    try:
        data_df.columns = ['TeamsRaw', 'Bookmaker', 'OddHome','OddDraw', 'OddAway', 'DateRaw', 'ScoreRaw']
    except:
        print('Function crashed, probable reason : no games scraped (empty season)')
        return(1)

    data_df["ScoreRaw"] = '0:0'
    ##################### FINALLY WE CLEAN THE DATA AND SAVE IT ##########################
    '''Now we simply need to split team names, transform date, split score'''

    # (0) Filter out None rows
    data_df = data_df[~data_df['Bookmaker'].isnull()].dropna().reset_index()
    data_df["TO_KEEP"] = 1
    for i in range(len(data_df["TO_KEEP"])):
        if len(re.split(':',data_df["ScoreRaw"][i]))<2 :
            data_df["TO_KEEP"].iloc[i] = 0

    data_df = data_df[data_df["TO_KEEP"] == 1]

    # (a) Split team names
    data_df["Home_id"] = [re.split(' - ',y)[0] for y in data_df["TeamsRaw"]]
    data_df["Away_id"] = [re.split(' - ',y)[1] for y in data_df["TeamsRaw"]]

    # (b) Transform date
    data_df["Date"] = [re.split(', ',y)[1] for y in data_df["DateRaw"]]

    # (c) Split score
    data_df["Score_home"] = 0
    data_df["Score_away"] = 0

    # (d) Set season column
    data_df["Season"] = SEASON


    # Finally we save results
    if not os.path.exists('./{}'.format(tournament)):
        os.makedirs('./{}'.format(tournament))
    data_df[['Home_id', 'Away_id', 'Bookmaker', 'OddHome','OddDraw', 'OddAway', 'Date', 'Score_home', 'Score_away','Season']].to_csv('./{}/NextGames_{}_{}.csv'.format(tournament,tournament, SEASON), sep=';', encoding='utf-8', index=False)


    return(data_df)

def reject_ads(switch_to_decimal = False):
    # Reject ads
    ffi2('//*[@id="onetrust-reject-all-handler"]')
    
    if switch_to_decimal:
        # Change odds to decimal format
        driver.find_element("xpath", '//*[@id="user-header-oddsformat-expander"]').click()
        driver.find_element("xpath", '//*[@id="user-header-oddsformat"]/li[1]/a/span').click()

def main_scraping_controller():
    country = 'spain'
    tournament = 'laliga'
    season = '2021-2022'
    max_page = 25
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    #chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(executable_path = DRIVER_LOCATION, chrome_options=chrome_options)
    df = scrape_league(
        driver=driver,
        Season=season,
        nseason=1,
        tournament1=tournament,
        sport="football",
        country1=country,
        max_page=max_page
    )
    driver.close()

if __name__ == '__main__':
    main_scraping_controller()
