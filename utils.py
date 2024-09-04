from bs4 import BeautifulSoup
import requests


def check_for_downloadable_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    forms = soup.find_all('a')

    return len(forms)