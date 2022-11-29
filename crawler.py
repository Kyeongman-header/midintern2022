import requests
from bs4 import BeautifulSoup as bs
import cloudscraper
from tqdm import trange
import time


def page_scrapper(n,category):
        print("page : " + str(n+1) + " category : " + category)
        scrap=cloudscraper.create_scraper()
        page=scrap.get("https://blog.reedsy.com/short-stories/"+category+"/page/"+str(n+1)+"/")

        #page=scrap.get("https://www.bookrix.com/books;fiction,id:16,page:"+str(n+1)+'.html') # 1페이지부터 시작하니까.
        soup=bs(page.text,"html.parser")
	# print(soup)
        links=soup.select('.mimic-h4 > .no-decoration')
        
        
        # print(links)
        if len(links)==0:
                print("eop")
                return "end of pages"
        
        contents=[]

        for l in links:
                
                inner_page=scrap.get("https://blog.reedsy.com/"+l.attrs['href'])
                soup=bs(inner_page.text,"html.parser")
                words=soup.select('.content-thin > article')
                one_novel=""
                for w in words:
                        one_novel=one_novel+w.text
                
                one_novel=one_novel.replace("⁂","")
                contents.append(one_novel)

                time.sleep(1)
        
        return contents

                # print(one_novel)
                
        #                 read_link="https://www.bookrix.com/"+soup.select_one('.button')['href']
        #                 
        #                 if (int(words[1].text.split(' ')[0]) <= 5000):
        #                         print(str(l) +" is not over 5000. it has :" + str(int(words[1].text.split(' ')[0])))
        #                         continue
        #                 else:
        #                         print(str(l) + " is downloading!")
        #         except:
        #                 print("error on the page.")
        #                 continue
        #         # print(read_link)
        #         read_page=scrap.get(read_link)
        #         soup=bs(read_page.text,"html.parser")
        #         fictions=soup.select('#text>p')
        #         # print(text)
        #         one_novel=""
        #         for t in fictions:
        #                 one_novel=one_novel+" "+ t.text
        #         # print(one_novel)

        #         content=[]
        #         splitted_one_novel=one_novel.split(".")
        #         # print(splitted_one_novel)
        #         # print(len(splitted_one_novel))
        #         # print(splitted_one_novel[0])
        #         # print("courtesy" in splitted_one_novel[0])
        #         for s in splitted_one_novel:
        #                 # print(s)
        #                 # input()
        #                 s=s.replace("\n","")
        #                 s=s.replace('\\',"")
        #                 s=s.replace('\xa0',"")
        #                 if "Imprint" in s:
        #                         continue
        #                 if "Google" in s:
        #                         continue
        #                 if "eBooks" in s or "ebook" in s or "ebooks" in s or "eBook" in s :
        #                         continue
        #                 if "COPYRIGHT" in s or "copyright" in s or "Copyright" in s:
        #                         continue
        #                 if "Cover photograph" in s:
        #                         continue
                        
        #                 if "net" in s or "org" in s:
        #                         continue
        #                 if "html" in s or "http" in s :
        #                         continue
        #                 if 'this permission is not passed onto others' in s:
        #                         continue
        #                 if "Publication Date:" in s:
        #                         break
        #                 if "EPILOGUE" in s:
        #                         break
        #                 if "AFTERWORD" in s:
        #                         break
        #                 if "Continued" in s:
        #                         break
        #                 if s == '':
        #                         continue
        #                 # print(s)
        #                 # print("This" in s and "book" in s and "was" in s and "distributed" in s and "courtesy" in s and "of:" in s)
        #                 content.append(s)

        #         # print(content)
        #         contents.append(content)
        # return contents

import csv
import tqdm
import others

# print(len(page_scrapper(1,"Adventure")))
scrap=cloudscraper.create_scraper()
page=scrap.get("https://blog.reedsy.com/short-stories/")
soup=bs(page.text,"html.parser")
cs=soup.select('.space-top-xs-md.space-bottom-xs-md > .space-bottom-xs-sm > a')
categories=[]
begin=False
for c in cs:
        if c.attrs['href'].split('/')[2] == 'mystery' or begin==True:
            begin=True
            categories.append(c.attrs['href'].split('/')[2])
        else:
            continue


file="reedsy_wp_5"
f=open(file+".csv",'w',encoding='utf-8',newline='')
wr=csv.writer(f)
whole_seq_length=[]
count=1
MAX_PAGE=500
begin = False
for c in categories:
        for i in range(MAX_PAGE):
                if begin or i > 88:
                    contents=page_scrapper(i,c)
                    begin=True
                    time.sleep(1)
                else:
                    continue

                if contents=="end of pages":
                        break
                else:
                        for t in range(len(contents)):
                                wr.writerow([count,contents[t]])
                                count=count+1
                        

# WHOLE_PAGE=4492
# f=open('bookrix_fictions.csv','w',encoding='utf-8',newline='')
# wr=csv.writer(f)
# whole_seq_length=[]
# count=1
# # #wr.writerow(['index','summary','fictions','Sequence_length'])

# for i in trange(WHOLE_PAGE):
#         contents=page_scrapper(i)
#         print(len(contents))
#         for t in range(len(contents)):
#                 wr.writerow([count,contents[t]])
#                 count=count+1
# f.close()

#f = open('tor_fictions.csv', 'r', encoding='utf-8')
#rdr = csv.reader(f)
# print(rdr)
#for line in rdr:
#    print(line[1])
#    print(line[2])
#    input()

f.close()    
