import requests
from bs4 import BeautifulSoup as bs
import cloudscraper

scrap=cloudscraper.create_scraper()
page=scrap.get("https://www.tor.com/category/all-fiction/page/4/")
soup=bs(page.text,"html.parser")
# print(soup)
elements=soup.select_one('#infinite-scroll-wrapper')
# print(elements)
elements=elements.select('header > h2 > a')

for index, element in enumerate(elements, 1):
		print("{} 번째 게시글: {}, {}".format(index, element.text, element.attrs['href']))
		inner_page=scrap.get(element.attrs['href'])
		soup=bs(inner_page.text,"html.parser")
		print(soup)
		# elements=soup.select_one('#post-365620 > div')
		# print(elements)
		# elements=elements.select('p')
		# print(elements)
