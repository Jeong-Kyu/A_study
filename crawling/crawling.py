import requests
from bs4 import BeautifulSoup

webpage = requests.get("https://www.daangn.com/hot_articles")
soup = BeautifulSoup(webpage.content, "html.parser")
print(soup.p, "\n") # p 태그 출력 
print(soup.p.string, "\n") # p 태그 출력(문자) 
print(soup.h1, "\n") # h1 태그 출력 


for child in soup.ul.children:
    print(child) # 하위항목

for parent in soup.ul.parents:
    print(parent) # 상위항목

print(soup.find_all("h2")) # h2 태그 모두찾기

soup.find_all(attrs={'class':'footer-list', 'id':'footer-address-list'}) # html 이용 필요한 부분 가져오기

def search_function(tag): #함수형
    return tag.attr('class') == "card-title" and tag.string == "Hello World"
soup.find_all(search_function)


#CSS 탐색
soup.select(".card-region-name") # class -> .
soup.select("#hot-articles-go-download") # id -> #
for x in range(0,10):
    print(soup.select(".card-title")[x].get_text()) # text 가져오기