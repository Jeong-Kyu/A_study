import requests
from bs4 import BeautifulSoup
from indeed import indeed_page

max_page = indeed_page()

for n in range(max_page):
    print(f"start = {n*50}")