---
layout: post
title:  "Draft Scraper"
date:   2023-02-21 01:08:51 +0000
categories: jekyll update
---
This is a small step in a larger goal I'm trying to get done.  The code is just a simple web scraper that collects draft information for a set amount of years.  Further on, I'd like to build a model to predict the round a college football player will get drafted in using this data combined with college statistics for the players.  The scraper gets the data from [pro-football-reference](https://www.pro-football-reference.com) where they have a bunch of data on the round that a player goes in, in addition to some of their professional stats.  I use selenium + chrome in order to scrape the data.  It's not especially fast, but performance improves a bit if you run the scraper in headless mode.  To get started, I had to get the correct driver for selenium to interact with the browser at the [chromedriver](https://chromedriver.chromium.org/downloads) site.  It just has to match up with the chrome version that I was using.  


Let's get into the necessary imports.  

{% highlight python %}
# imports
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from typing import *

import pandas as pd
import json
import os
import pdb
{% endhighlight %}

Generally, when I build a scraper, I like using WebDriverWait + expected_conditions, which works well on websites that have to load a ton of elements or popups.  I also prefer to find page elements via XPATH when its convenient, since I've found its slightly faster than searching by tag or other means.  I also have a linux machine that I also develop on so I had a few versions of the chromedriver saved, hence the `DRIVERNAME`.  

{% highlight python %}
# vars
MAC = False
CWD = os.getcwd()
DRIVERNAME = "/chromedriver_mac" if MAC else "/chromedriver_linux"
DRIVERPATH = CWD + DRIVERNAME
HEADLESS = False
START_YEAR = 2007
YEARS = 7
DRAFT_TABLE_XPATH = "//div[@id='div_drafts']/table/tbody"
COMBINE_TABLE_XPATH = "//div[@id='div_combine']/table/tbody"
DRAFTPAGE = f"https://www.pro-football-reference.com/years/{START_YEAR}/draft.htm"
COMBINEPAGE = f"https://www.pro-football-reference.com/draft/{START_YEAR}-combine.htm"

# options for chrome
OPTIONS = ChromeOptions()
OPTIONS.add_argument("--enable-javascript")
OPTIONS.add_argument("--disable-blink-feature=AutomationControlled")
OPTIONS.add_argument("start-maximized")
OPTIONS.add_experimental_option("excludeSwitches", ["enable-automation"])

OPTIONS.headless = HEADLESS
{% endhighlight %}

The any arg with COMBINE in it refers to another scraper I'm working on to get combine statistics.  It uses many of the same functions that I created for the draft stats but it's still a work in progress.  

The scraper works in a few different steps.  First, it gets the table rows from the draft page and filters those rows down into just the players, ignoring the table headers except for the top one.  Then, if it hasn't been created, it makes a dictionary with the table headers as keys.  It iterates through each player and adds their stats to the created dictionary in the appropriate list.  The process continues until the specified number of years is reached.  

Here are the functions that I use in the scraper:

{% highlight python %}
def make_draft_dict(player1):
    """
    params
    ---
    player1 : selenium.webdriver.remote.webelement.WebElement
    first player in the table, just to make the dict for the data

    ret
    ---
    draft_data : dict
    empty dict with the right titles
    """
    tds = player1.find_elements(By.XPATH, "//td")[:27]
    draft_data = {}

    draft_data["draft_round"] = []
    for td in tds:
        header = td.get_attribute("data-stat")
        draft_data[header] = []

    return draft_data

def make_combine_dict(player1):
    ths = player1.find_elements(By.XPATH, "//th")[:13]
    combine_data = {}

    for th in ths:
        header = th.get_attribute("data-stat")
        combine_data[header] = []

    return combine_data

def get_tablerows(driver, table_xpath: str):
    """
    params
    ---
    driver : selenium.webdriver
    table_xpath : str

    ret
    --- 
    trs : List[selenium.webdriver.remote.webelement.WebElement]
    rows of the table
    """

    cond = ec.presence_of_element_located((By.XPATH, table_xpath))
    tb = WebDriverWait(driver, 20).until(cond)
    trs = tb.find_elements(By.XPATH, "//tr")

    # first element is the 
    return trs[2:]

# start at 2, and only get the ones that don't have a class attribute
def filter_rows(trs):
    """
    params
    ---
    trs : selenium.webdriver.remote.webelement.WebElement
    
    ret
    ---
    players: List[str]
    list of players stats
    """

    players = [] 
    for tr in trs:
        # empty class means its not a header row
        if not tr.get_attribute("class") and tr.text:
            players.append(tr)
    
    return players

def fill_player_data(player, draft_data: dict):
    """
    params
    ---
    player : selenium.webdriver.remote.webelement.WebElement
    
    ret
    ---
    draft_data : dict
    update dict with that player's data
    """
    rnd = player.find_element(By.TAG_NAME, "th").text
    
    draft_data["draft_round"].append(rnd)
    keys = list(draft_data.keys())
    keys.remove("draft_round")

    data = player.find_elements(By.TAG_NAME, "td")
    for i, key in enumerate(keys):
        stat = data[i].text
        draft_data[key].append(stat)
    return draft_data

def fill_player_combine_data(player, combine_data: dict):
    playername = player.find_element(By.TAG_NAME, "th").text
    combine_data["player"].append(playername)
    
    keys = list(combine_data.keys())
    keys.remove("player")

    data = player.find_elements(By.TAG_NAME, "td")
    for i, key in enumerate(keys):
        stat = data[i].text
        combine_data[key].append(stat)

    return combine_data

def go_next_year(last_page, prev_year, next_year, driver):
    """
    params
    ---
    last_page : str
    the url of the current page

    prev_year : str
    year of the current page

    next_year : str
    the next year to scrape

    driver : selenium.webdriver

    ret
    ---
    driver : selenium.webdriver
    """

    new_page = last_page.replace(prev_year, next_year)
    driver.get(new_page)
    return driver, new_page
{% endhighlight %}

And finally, here's the code to run the scraper:

{% highlight python %}
  def scrape_draft_data(numyears: int = YEARS, starting_url: str = DRAFTPAGE):
    driver = webdriver.Chrome(executable_path = DRIVERPATH, options = OPTIONS)
    driver.get(starting_url)
    last_page = starting_url
    next_year = START_YEAR

    for year in range(1, numyears + 1):
        prev_year = next_year
        trs = get_tablerows(driver, DRAFT_TABLE_XPATH)
        players = filter_rows(trs)

        # do this once
        if year == 1:
            draft_data = make_draft_dict(players[0])
        for player in players:
            print(player.text)
            draft_data = fill_player_data(player, draft_data)

        next_year = prev_year - 1
        driver, last_page = go_next_year(last_page, str(prev_year), str(next_year), driver)

    return draft_data
{% endhighlight %}

Pretty quick and simple!  With this code, I scraped draft data for the past 20 years and plan to do the same for combine data and college stats data as well.  Thanks for reading!