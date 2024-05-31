import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By



country_code = {
    "DZA": "Algeria",
    "ARG": "Argentina",
    "AUS": "Australia",
    "AUT": "Austria",
    "BEL": "Belgium",
    "BRA": "Brazil",
    "CHL": "Chile",
    "CHN": "China PR",
    "DNK": "Denmark",
    "ECU": "Ecuador",
    "ENG": "England",
    "FIN": "Finland",
    "FRA": "France",
    "GER": "Germany",
    "GRE": "Greece",
    "ISR": "Israel",
    "ITA": "Italy",
    "JPN": "Japan",
    "KOR": "Korea Republic",
    "MEX": "Mexico",
    "MAR": "Morocco",
    "HOL": "Netherlands",
    "NZL": "New Zealand",
    "NOR": "Norway",
    "POR": "Portugal",
    "RUS": "Russia",
    "SCO": "Scotland",
    "ZAF": "South Africa",
    "SPA": "Spain",
    "SWE": "Sweden",
    "CHE": "Switzerland",
    "TUN": "Tunisia",
    "USA": "USA",
    "VEN": "Venezuela"
}


country_code_oddsportal = {
    "DZA": "Algeria",
    "ARG": "Argentina",
    "AUS": "Australia",
    "AUT": "Austria",
    "BEL": "Belgium",
    "BRA": "Brazil",
    "CHL": "Chile",
    "CHN": "China",
    "DNK": "Denmark",
    "ECU": "Ecuador",
    "ENG": "England",
    "FIN": "Finland",
    "FRA": "France",
    "GER": "Germany",
    "GRE": "Greece",
    "ISR": "Israel",
    "ITA": "Italy",
    "JPN": "Japan",
    "KOR": "South Korea",
    "MEX": "Mexico",
    "MAR": "Morocco",
    "HOL": "Netherlands",
    "NZL": "New Zealand",
    "NOR": "Norway",
    "POR": "Portugal",
    "RUS": "Russia",
    "SCO": "Scotland",
    "ZAF": "South Africa",
    "SPA": "Spain",
    "SWE": "Sweden",
    "CHE": "Switzerland",
    "TUN": "Tunisia",
    "USA": "USA",
    "VEN": "Venezuela"
}

league_map_oddsportal = {
    'DZA1': "Ligue 1",
    'ARG1': "Liga Profesional",
    'AUS1': "A-League",
    'AUT1': "Bundesliga",
    'BEL1': "Jupiler Pro League",
    'BRA1': "Serie A",
    'BRA2': "Serie B",
    'CHL1': "Primera Division",
    'CHN1': "Super League",
    'DNK1': "1st Division",
    'ECU1': "Liga Pro",
    'ENG1': "Premier League",
    'ENG2': "Championship",
    'ENG3': "League One",
    'ENG4': "League Two",
    'ENG5': "National League",
    'FIN1': "Veikkausliiga",
    'FRA1': "League 1",
    'FRA2': "League 2",
    'FRA3': "National",
    'GER1': "Bundesliga",
    'GER2': "2. Bundesliga",
    'GER3': "3. Liga",
    'GRE1': "Super League",
    'ISR1': "Ligat ha Al",
    'ITA1': "Serie A",
    'ITA2': "Serie B",
    'JPN1': "J1 League",
    'JPN2': "J2 League",
    'KOR1': "K League 1",
    'MEX1': "Liga MX",
    'MAR1': "Botalo Pro",
    'HOL1': "Eredivisie",
    'NZL1': "National League",
    'NOR1': "Eliteserien",
    'NOR2': "OBOS-ligaen",
    'POR1': "Liga Portugal",
    'RUS1': "Premier League",
    'RUS2': "FNL",
    'SCO1': "Premiership",
    'SCO2': "Championship",
    'SCO3': "League One",
    'SCO4': "League Two",
    'ZAF1': "Premier League",
    'SPA1': "LaLiga",
    'SWE1': "Allsvenskan",
    'CHE1': "Super League",
    'TUN1': "Ligue Professionnelle 1",
    'USA1': "MLS",
    'USA2': "USL Championship",
    'VEN1': "Primera Division"
}

URL = "https://www.oddsportal.com/soccer/germany/bundesliga-2021-2022/results/"


web = "https://www.oddsportal.com/site-map-active/"
path = "C:\\Users\\ke6564\\Downloads\\chromedriver_win32\\chromedriver.exe"

driver = webdriver.Chrome()
driver.get(web)
accept = driver.find_element(by=By.XPATH, value='/html/body/div[3]/div[2]/div/div[1]/div/div[2]/div/button[1]')
accept.click()


links = driver.find_elements(By.XPATH, "//a[contains(@href, '/soccer/')]")
links_text = [x.get_attribute("href") for x in links]

links_text = list(set(links_text))
links_text = [x.replace("https://www.oddsportal.com/soccer/", "") for x in links_text]
links_text = [x.split("/")[:-1] for x in links_text]
links_text = [x for x in links_text if len(x) > 1]

links_text.sort(key=lambda x: (x[0], x[1]))

countries = list(country_code_oddsportal.values())
countries = [x.lower().replace(" ", "-") for x in countries]

leagues = list(league_map_oddsportal.values())
leagues = [x.lower().replace(" ", "-") for x in leagues]


country_links = [x for x in links_text if x[0] in countries and x[1] in leagues]
country_links = [x for x in links_text if x[1] in leagues]
len(country_links)

x = set([x[0] for x in country_links])
[y for y in countries if y not in x]




