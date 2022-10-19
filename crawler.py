import requests
from bs4 import BeautifulSoup as bs
import cloudscraper


def page_scrapper(n):
	scrap=cloudscraper.create_scraper()
	page=scrap.get("https://www.tor.com/category/all-fiction/page/"+str(n)+'/')
	soup=bs(page.text,"html.parser")
	# print(soup)
	elements=soup.select_one('#infinite-scroll-wrapper')
	# print(elements)
	elements=elements.select('header > h2 > a')
	texts=[]
	seq_lengths=[]
	for index, element in enumerate(elements, 1):
			print("{} 번째 게시글: {}, {}".format(index, element.text, element.attrs['href']))
			inner_page=scrap.get(element.attrs['href'])
			soup=bs(inner_page.text,"html.parser")
			# print(soup)
			e=soup.select_one('#content')
			e=e.select('div.entry-content > p')
			# print(e)
			l=len(e)
			seq_lengths.append(l)
			text=""
			for index, element in enumerate(e, 1):
				if index < l-1:
					# print("{}".format(element.text))
					text=text+" " + element.text
			# print(text)
			
			texts.append(text)
	return texts,seq_lengths


import csv
import tqdm
import others

f=open('tor_fictions.csv','w',encoding='utf-8',newline='')
wr=csv.writer(f)
whole_seq_length=[]
count=1
wr.writerow(['index','fictions','Sequence_length'])

for i in range(1):
	texts,seq_lengths=page_scrapper(i)
	for t in range(len(texts)):
		wr.writerow([count,texts[t],seq_lengths[t]])
		count=count+1
f.close()

# f = open('tor_fictions.csv', 'r', encoding='utf-8')
# rdr = csv.reader(f)
# # print(rdr)
# for line in rdr:
# 	print(line[0])
	
# 	input()

# f.close()    