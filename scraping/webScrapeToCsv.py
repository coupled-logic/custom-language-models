import csv
import requests
from bs4 import BeautifulSoup

url = 'https://wiki.gunthy.org/'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
results = soup.find(id='aside')
menu_links = results.find_all('a')
file = open('gunbot_wiki_links.csv', 'w')
writer = csv.writer(file)
# write header rows #
writer.writerow(['topic', 'url'])

for menu_link in menu_links:
    topic = menu_link.text.strip()
    url = menu_link['href']
    writer.writerow([topic.encode('utf-8'), url.encode('utf-8')])

file.close()
