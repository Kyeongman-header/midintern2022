import requests
from bs4 import BeautifulSoup as bs
import cloudscraper
from tqdm import trange


def page_scrapper(n):
        scrap=cloudscraper.create_scraper()
        page=scrap.get("https://www.bookrix.com/books;fiction,id:16,page:"+str(n+1)+'.html') # 1페이지부터 시작하니까.
        soup=bs(page.text,"html.parser")
	# print(soup)
        links=soup.select('.item-title>a')
        
        
        # print(links)
        contents=[]

        for l in links:
                # input()
                try:
                        inner_page=scrap.get("https://www.bookrix.com/"+l.attrs['href'])
                        soup=bs(inner_page.text,"html.parser")
                
                        read_link="https://www.bookrix.com/"+soup.select_one('.button')['href']
                        words=soup.select('.navbar>li')
                        if (int(words[1].text.split(' ')[0]) <= 5000):
                                print(str(l) +" is not over 5000. it has :" + str(int(words[1].text.split(' ')[0])))
                                continue
                        else:
                                print(str(l) + " is downloading!")
                except:
                        print("error on the page.")
                        continue
                # print(read_link)
                read_page=scrap.get(read_link)
                soup=bs(read_page.text,"html.parser")
                fictions=soup.select('#text>p')
                # print(text)
                one_novel=""
                for t in fictions:
                        one_novel=one_novel+" "+ t.text
                # print(one_novel)

                content=[]
                splitted_one_novel=one_novel.split(".")
                # print(splitted_one_novel)
                # print(len(splitted_one_novel))
                # print(splitted_one_novel[0])
                # print("courtesy" in splitted_one_novel[0])
                for s in splitted_one_novel:
                        # print(s)
                        # input()
                        s=s.replace("\n","")
                        s=s.replace('\\',"")
                        s=s.replace('\xa0',"")
                        if "Imprint" in s:
                                continue
                        if "Google" in s:
                                continue
                        if "eBooks" in s or "ebook" in s or "ebooks" in s or "eBook" in s :
                                continue
                        if "COPYRIGHT" in s or "copyright" in s or "Copyright" in s:
                                continue
                        if "Cover photograph" in s:
                                continue
                        
                        if "net" in s or "org" in s:
                                continue
                        if "html" in s or "http" in s :
                                continue
                        if 'this permission is not passed onto others' in s:
                                continue
                        if "Publication Date:" in s:
                                break
                        if "EPILOGUE" in s:
                                break
                        if "AFTERWORD" in s:
                                break
                        if "Continued" in s:
                                break
                        if s == '':
                                continue
                        # print(s)
                        # print("This" in s and "book" in s and "was" in s and "distributed" in s and "courtesy" in s and "of:" in s)
                        content.append(s)

                # print(content)
                contents.append(content)
        return contents

import csv
import tqdm
import others

page_scrapper(1)

WHOLE_PAGE=4492
f=open('bookrix_fictions.csv','w',encoding='utf-8',newline='')
wr=csv.writer(f)
whole_seq_length=[]
count=1
# #wr.writerow(['index','summary','fictions','Sequence_length'])

for i in trange(WHOLE_PAGE):
        contents=page_scrapper(i)
        print(len(contents))
        for t in range(len(contents)):
                wr.writerow([count,contents[t]])
                count=count+1
f.close()

#f = open('tor_fictions.csv', 'r', encoding='utf-8')
#rdr = csv.reader(f)
# print(rdr)
#for line in rdr:
#    print(line[1])
#    print(line[2])
#    input()

#f.close()    
