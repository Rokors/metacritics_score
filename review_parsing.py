# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:44:44 2020

@author: barkov
"""
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import csv

def review_text(body): 
    soup = BeautifulSoup(body, 'html.parser')
    text = []
    for div in soup.findAll('div',{'class':'review_body'}):
        text.append(div.text.strip())
    return text
    #return u" ".join(t.strip() for t in text)

#number of links to parse for each category
LINKNUM = 1000
# get list of urls for chosen game params

# load dataset
pbl = pd.read_table('complete_base.csv')
pbl.dropna(subset = ['UserReviews', 'CriticReviews', 'MetaScore', 'UserScore','Publisher', 'Developer'], inplace=True)
pbl = pbl.drop(pbl[pbl.CriticReviews < 5].index)
pbl = pbl.drop(pbl[pbl.UserReviews < 15].index)
list_space = []
for _ in range(len(pbl['Title'])):
    list_space.append('/critic-reviews')
pbl['space'] = list_space
pbl['reviewlink'] = pbl.permalink + pbl.space


# first part, games >87 MetaScore

pbl_1 = pbl.drop(pbl[pbl.MetaScore < 87].index)
review_links = pbl_1.reviewlink.tolist()
#review_links = review_links[:LINKNUM]

user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers={'User-Agent':user_agent,} 

list_text = []
for link in review_links:
    request=urllib.request.Request(link,None,headers) #The assembled request
    try:
        response = urllib.request.urlopen(request)
    except Exception:
        pass
    else:
        data = response.read() # The data u need
    #html = urllib.request.urlopen(link,headers=hdr).read()
        list_text.append(review_text(data))
    
best_text = []
for sublist in list_text:
    for item in sublist:
        best_text.append(item)
del list_text
with open('best_text.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(best_text) 
# second part, games <90 && >75 MetaScore

pbl_1 = pbl.drop(pbl[pbl.MetaScore > 87].index)
pbl_1 = pbl.drop(pbl[pbl.MetaScore < 70].index)


review_links = pbl_1.reviewlink.tolist()
review_links = review_links[:LINKNUM]



list_text = []
for link in review_links:
    request=urllib.request.Request(link,None,headers) #The assembled request
    try:
        response = urllib.request.urlopen(request)
    except Exception:
        pass
    else:
        data = response.read() # The data u need
    #html = urllib.request.urlopen(link,headers=hdr).read()
        list_text.append(review_text(data))
    
average_text = []
for sublist in list_text:
    for item in sublist:
        average_text.append(item)
del list_text
with open('average_text.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(average_text)
# third part, games <75 MetaScore

pbl_1 = pbl.drop(pbl[pbl.MetaScore > 65].index)


review_links = pbl_1.reviewlink.tolist()
review_links = review_links[:LINKNUM]


list_text = []
for link in review_links:
    request=urllib.request.Request(link,None,headers) #The assembled request
    try:
        response = urllib.request.urlopen(request)
    except Exception:
        pass
    else:
        data = response.read() # The data u need
    #html = urllib.request.urlopen(link,headers=hdr).read()
        list_text.append(review_text(data))
    
bad_text = []
for sublist in list_text:
    for item in sublist:
        bad_text.append(item)
del list_text

with open('bad_text.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(bad_text)
   
