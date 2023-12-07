from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
import time
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from requests import get
from bs4 import BeautifulSoup
import time
import pandas as pd

def get_links(page):
    # each page contains 120 posts' links
    for i in range(25, min(145,len(page))):
        link = page[i].get_attribute("href")
        if link == "":
            return "all pages are done"
        post_link_list.append(link)
    return "One page is done"

# URL to start scrape (first page) 
url = "https://lasvegas.craigslist.org/search/goldfield-nv/ccc?lat=37.2808&lon=-117.393&search_distance=260#search=1~list~0~0"
#url = 'https://tippecanoe.craigslist.org/search/templeton-in/mis?lat=40.6167&lon=-87.3209&search_distance=1000#search=1~list~0~0'
post_link_list = []

# Start the Chrome browser
with webdriver.Chrome() as driver:
    driver.get(url)
    # Wait for the page to load
    time.sleep(2)
    count = 0
    while True:
        # Get the macro-container for the posts
        posts = driver.find_elements(By.TAG_NAME, 'a')
        executting_result = get_links(posts)
        if executting_result == "all pages are done":
            print("All pages have been scraped")
            break
        # Check if there is a button to go to the next page
        next_button = driver.find_element(By.CLASS_NAME, 'cl-next-page')  # Adjust the selector based on the actual class of the next button
        if 'disabled' in next_button.get_attribute('class'):
            print("All pages have been scraped")
            break
        # Click the next page button
        time.sleep(2.5)
        driver.execute_script("arguments[0].click();", next_button)
        time.sleep(2.5)
        count += 1
        # enter the last page to stop loop
        if count == 50:
            break

# Create a DataFrame and save to CSV
df = pd.DataFrame(post_link_list, columns=["link"])
df.to_csv('links.csv')


# Open the CSV file for reading
# enter the file path of the file containing all posts' links
csv_file_path = 'links.csv'
df = pd.read_csv(csv_file_path)
df = df.rename(columns={df.columns[0]: "index"})

def get_content(link, max_retries=2, retry_interval=1):
    for _ in range(max_retries):
        response = get(link)
        time.sleep(retry_interval)
        
        if response.status_code == 200:
            html_soup = BeautifulSoup(response.text, 'html.parser')
            title = html_soup.find("h1").text
            content = html_soup.find("section", id="postingbody").text
            return title, content
        
    # Return empty strings if the maximum number of retries is reached
    return "", ""

titles = []
contents = []
# use a for loop to scrape the title and content in each post
for i in range(len(df['link'])):
    title, content = get_content(df.iloc[i, 1])
    print(f"The row {df.iloc[i, 0]} is done.")
    content = content.replace("QR Code Link to This Post", "")
    titles.append(title.strip())
    contents.append(content.strip())

df["title"] = titles + [ "" for _ in range(len(df) - len(titles))]
df["content"] = contents + [ "" for _ in range(len(df) - len(titles))]

# output the result as a csv file
df.to_csv("contents.csv", index=False, columns= ["index", "link", "title", "content"])
print("scraping is done")