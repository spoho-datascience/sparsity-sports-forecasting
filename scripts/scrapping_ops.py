import click
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
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC


from selenium.webdriver.common.action_chains import ActionChains

#from create_clean_table import *

DRIVER_LOCATION = "/opt/chromedriver"

TOURNAMENTS_OP = {
    'DZA1': {
        "formal": "Ligue 1",
        "link": "ligue-1",
        "country": "algeria"
    },
    'ARG1': {
        "formal": "Liga Profesional",
        "link": "liga-profesional",
        "other_link": "primera-division",
        "country": "argentina"
    },
    'AUS1': {
        "formal":"A-League",
        "link": "a-league",
        "country": "australia"
    },
    'AUT1': {
        "formal":"Bundesliga",
        "link": "bundesliga",
        "other_link": "t-mobile-bundesliga",
        "country": "austria"
    },
    'BEL1': {
        "formal":"Jupiler Pro League",
        "link": "jupiler-pro-league",
        "other_link": "jupiler-league",
        "country": "belgium"
    },
    'BRA1': {
        "formal":"Serie A",
        "link": "serie-a",
        "country": "brazil"
    },
    'BRA2': {
        "formal":"Serie B",
        "link": "serie-b",
        "country": "brazil"
    },
    'CHL1': {
        "formal":"Primera Division",
        "link": "primera-division",
        "country": "chile"
    },
    'CHN1': {
        "formal":"Super League",
        "link": "super-league",
        "country": "china"

    },
    'DNK1': {
        "formal":"1st Division",
        "link": "1st-division",
        "other_link": "nordicbet-ligaen",
        "country": "denmark"
    },
    'ECU1': {
        "formal":"Liga Pro",
        "link": "liga-pro",
        "other_link": "serie-a",
        "country": "ecuador"
    },
    'ENG1': {
        "formal":"Premier League",
        "link": "premier-league",
        "country": "england"
    },
    'ENG2': {
        "formal":"Championship",
        "link": "championship",
        "country": "england"
    },
    'ENG3': {
        "formal":"League One",
        "link": "league-one",
        "country": "england"
    },
    'ENG4': {
        "formal":"League Two",
        "link": "league-two",
        "country": "england"
    },
    'ENG5': {
        "formal":"National League",
        "link": "national-league",
        "other_link": "nationwide-conference",
        "country": "england"
    },
    'FIN1': {
        "formal":"Veikkausliiga",
        "link": "veikkausliiga",
        "country": "finland"
    },
    'FRA1': {
        "formal":"Ligue 1",
        "link": "ligue-1",
        "other_link": "division-1",
        "country": "france"
    },
    'FRA2': {
        "formal":"Ligue 2",
        "link": "ligue-2",
        "country": "france"
    },
    'FRA3': {
        "formal":"National",
        "link": "national",
        "country": "france"
    },
    'GER1': {
        "formal":"Bundesliga",
        "link": "bundesliga",
        "country": "germany"
    },
    'GER2': {
        "formal":"2. Bundesliga",
        "link": "2-bundesliga",
        "country": "germany"
    },
    'GER3': {
        "formal":"3. Liga",
        "link": "3-liga",
        "country": "germany"
    },
    'GRE1': {
        "formal":"Super League",
        "link": "super-league",
        "country": "greece"
    },
    'ISR1': {
        "formal":"Ligat ha Al",
        "link": "ligat-ha-al",
        "country": "israel"
    },
    'ITA1': {
        "formal":"Serie A",
        "link": "serie-a",
        "country": "italy"
    },
    'ITA2': {
        "formal":"Serie B",
        "link": "serie-b",
        "country": "italy"
    },
    'JPN1': {
        "formal":"J1 League",
        "link": "j1-league",
        "other_link": "j-league",
        "country": "japan"
    },
    'JPN2': {
        "formal":"J2 League",
        "link": "j2-league",
        "other_link": "j-league-division-2",
        "country": "japan"
    },
    'KOR1': {
        "formal":"K League 1",
        "link": "k-league-1",
        "other_link": "k-league",
        "country": "south-korea"
    },
    'MEX1': {
        "formal":"Liga MX",
        "link": "liga-mx",
        "other_link": "primera-division",
        "country": "mexico"
    },
    'MAR1': {
        "formal":"Botalo Pro",
        "link": "botola-pro",
        "other_link": "botola",
        "country": "morocco"
    },
    'HOL1': {
        "formal":"Eredivisie",
        "link": "eredivisie",
        "country": "netherlands"
    },
    'NZL1': {
        "formal":"National League",
        "link": "national-league",
        "country": "new-zealand"
    },
    'NOR1': {
        "formal":"Eliteserien",
        "link": "eliteserien",
        "other_link": "tippeligaen",
        "country": "norway"
    },
    'NOR2': {
        "formal":"OBOS-ligaen",
        "link": "obos-ligaen",
        "other_link": "adeccoligaen",
        "country": "norway"
    },
    'POR1': {
        "formal":"Liga Portugal",
        "link": "liga-portugal",
        "other_link": "primeira-liga",
        "country": "portugal"
    },
    'RUS1': {
        "formal":"Premier League",
        "link": "premier-league",
        "country": "russia"
    },
    'RUS2': {
        "formal":"FNL",
        "link": "fnl",
        "other_link": "division-1",
        "country": "russia"
    },
    'SCO1': {
        "formal":"Premiership",
        "link": "premiership",
        "other_link": "premier-league",
        "country": "scotland"
    },
    'SCO2': {
        "formal":"Championship",
        "link": "championship",
        "other_link": "division-1",
        "country": "scotland"
    },
    'SCO3': {
        "formal":"League One",
        "link": "league-one",
        "other_link": "division-2",
        "country": "scotland"
    },
    'SCO4': {
        "formal":"League Two",
        "link": "league-two",
        "other_link": "division-3",
        "country": "scotland"
    },
    'ZAF1': {
        "formal":"Premier League",
        "link": "premier-league",
        "country": "south-africa"
    },
    'SPA1': {
        "formal":"LaLiga",
        "link": "laliga",
        "other_link": "primera-division",
        "country": "spain"
    },
    'SPA2': {
        "formal": "LaLiga2",
        "link": "laliga2",
        "other_link": "segunda-division",
        "country": "spain"
    },
    'SWE1': {
        "formal":"Allsvenskan",
        "link": "allsvenskan",
        "country": "sweden"
    },
    'CHE1': {
        "formal":"Super League",
        "link": "super-league",
        "country": "switzerland"
    },
    'TUN1': {
        "formal":"Ligue Professionnelle 1",
        "link": "ligue-professionnelle-1",
        "country": "tunisia"
    },
    'USA1': {
        "formal":"MLS",
        "link": "mls",
        "country": "usa"
    },
    'USA2': {
        "formal":"USL Championship",
        "link": "usl-championship",
        "other_link": "usl",
        "country": "usa"
    },
    'VEN1': {
        "formal":"Liga FUTVE",
        "link": "liga-futve",
        "other_link": "primera-division",
        "country": "venezuela"
    }
}

class OddsPortalScrapper:

    def __init__(self, **kwargs) -> None:
        chrome_options = Options()
        chrome_options.add_argument('--no-sandbox')
        if kwargs.get('headless', True):
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(executable_path = DRIVER_LOCATION, chrome_options=chrome_options)
        self.driver.maximize_window()
        self.odds_type = kwargs.get('odds_type', 'past')

    def fi2(self, a):
        try:
            self.driver.find_element("xpath", a).click()
        except:
            return False

    def ffi2(self, a):
        if self.fi2(a) != False :
            self.fi2(a)
            return(True)
        else:
            return(None)

    def reject_ads(self):
        # Reject ads
        self.ffi2('//*[@id="onetrust-reject-all-handler"]')

    def get_matches_links(self, link, link_regex="/football/spain/laliga-2021-2022/"):
        self.driver.get(link)
        self.reject_ads()
        time.sleep(1)
        links = WebDriverWait(self.driver, 5).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, f"a[href*='{link_regex}']")))
        while True:
            # Scroll down to last name in list
            self.driver.execute_script('arguments[0].scrollIntoView();', links[-1])
            try:
                # Wait for more names to be loaded
                WebDriverWait(self.driver, 5).until(lambda driver: len(WebDriverWait(driver, 5).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, f"a[href*='{link_regex}']")))) > len(links))
        
                # Update names list 
                links = WebDriverWait(self.driver, 5).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, f"a[href*='{link_regex}']")))
            except:
                # Break the loop in case no new names loaded after page scrolled down
                break
        action_links = [l.get_attribute('href') for l in links]
        match_action_links = []
        for action_link in action_links:
            if ('results' not in action_link) and ('standings' not in action_link) and ('outrights' not in action_link) and (not action_link.endswith(link_regex)) and (action_link not in match_action_links):
                match_action_links.append(action_link)
        click.echo(f"Found {len(match_action_links)} links to matches.")
        return match_action_links

    def get_match_info(self, match_link):
        self.driver.get(match_link)
        time.sleep(0.5)
        match_dict = {}
        try:
            self.reject_ads()
            odds_row = self.driver.find_element(
                by=By.XPATH, value="//*[text()='Average']"
            ).find_element(
                by=By.XPATH, value="../.."
            ).find_elements(
                by=By.XPATH, value=".//div"
            )
            for odds_name, odds_info in [('odds_home', odds_row[1]), ('odds_draw', odds_row[3]), ('odds_away', odds_row[5])]:
                match_dict[odds_name] = odds_info.text

            match_date = self.driver.find_element(
                by=By.CLASS_NAME, value="bg-event-start-time"
            ).find_elements(
                by=By.XPATH, value="..//div"
            )[1].text
            match_dict["match_date"] = match_date

            team_logo_images = self.driver.find_elements(
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
                match_dict[f"{home_away[team_i]}_team"] = team_name
                try:
                    score = int(results_divs[1].text)
                    match_dict[f"{home_away[team_i]}_score"] = score
                except:
                    pass
            
        except Exception as e:
            click.echo(str(e))
        return match_dict

    def scrape_season_page(self, page, country, tournament, season):
        if self.odds_type == 'past':
            link = 'https://www.oddsportal.com/football/{}/{}-{}/results/#/page/{}'.format(country,tournament,season,page) if season != "2022-2023" else 'https://www.oddsportal.com/football/{}/{}/results/#/page/{}'.format(country,tournament,page)
        else:
            link = 'https://www.oddsportal.com/football/{}/{}'.format(country,tournament)
        DATA = []
        click.echo(link)
        match_links = self.get_matches_links(link, link_regex=f"/football/{country}/{tournament}-{season}/" if season not in ['2022-2023', 'next'] else f"/football/{country}/{tournament}/" )
        for match in match_links:
            click.echo(match)
            DATA.append(self.get_match_info(match))
        return DATA

    def scrape_season_tournament(self, tournament, country, season, max_page, country_code):
        DATA_ALL = []
        for page in range(1, max_page + 1):
            data = self.scrape_season_page(page, country, tournament, season)
            DATA_ALL = DATA_ALL + [y for y in data if y != None]
        data_df = pd.DataFrame(DATA_ALL)
        data_df.to_csv(f'{country_code}-{country}-{tournament}-{season}-data.csv')

    def extract(self, seasons, country, tournament_name, max_page = 20, country_code=""):
        if self.odds_type == 'past':
            for s in seasons:
                print('We start to collect season {}'.format(s))
                self.scrape_season_tournament(tournament = tournament_name, country = country, season = s, max_page = max_page, country_code=country_code)
                print('We finished to collect season {} !'.format(s))
        elif self.odds_type == 'future':
            print('We start collecting future odds')
            self.scrape_season_tournament(tournament = tournament_name, country = country, season = '2022-2023', max_page = 1, country_code=country_code)
            print('We finished')

@click.group()
def cli():
    pass

@cli.command()
@click.argument('seasons', type=str, default='ALL')
@click.argument('tournaments', type=str, default='ALL')
@click.option('--other-link', '-o', 'other_link', is_flag=True, show_default=True, default=False)
@click.option('--large-season', '-l', 'large_season_format', is_flag=True, show_default=True, default=True)
@click.option('--show-browser', '-s', 'show_browser', is_flag=True, show_default=True, default=False)
@click.option('--next-matches', '-n', 'next_matches', is_flag=True, show_default=True, default=False)
def odds_portal(seasons, tournaments, other_link, large_season_format, show_browser, next_matches):
    link_pointer = 'link' if not other_link else 'other_link'
    if tournaments == 'ALL':
        tournaments_list = list(TOURNAMENTS_OP.keys())
        tournaments_list.remove('CHN1')
    else:
        tournaments_list = tournaments.split(",")
    if seasons == 'ALL':
        start_season = 2022
        seasons_list = []
        while start_season > 2000:
            seasons_list.append(f"{start_season - 1}-{start_season}" if large_season_format else f"{start_season}")
            start_season -= 1
    else:
        seasons_list = seasons.split(",")
    print(f"{len(seasons_list)} seasons: ", seasons_list)
    print(f"{len(tournaments_list)} tournaments: ", tournaments_list)
    op_scrapper = OddsPortalScrapper(headless=not show_browser, odds_type = 'past' if not next_matches else 'future')
    for tournament_key in tournaments_list:
        if tournament_key in TOURNAMENTS_OP:
            tournament_info = TOURNAMENTS_OP[tournament_key]
            country = tournament_info['country']
            tournament_link = tournament_info.get(link_pointer, tournament_info['link'])
            op_scrapper.extract(
                seasons=seasons_list, 
                country=country, 
                tournament_name=tournament_link,
                country_code=tournament_key
            )
        else:
            click.echo(f"Could not find tournament {tournament_key}")

    


if __name__ == '__main__':
    cli()
