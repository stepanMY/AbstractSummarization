from parsing.parser import AsyncParser
from bs4 import BeautifulSoup


def get_urls_from_page(url, text):
    """ Собирает все ссылки на курсовые со страницы """
    urls = []
    soup = BeautifulSoup(text, features="lxml")
    pres = soup.find_all('h3', {'class': 'vkr-card__title'})
    for pre in pres:
        url = 'https://www.hse.ru' + str(pre.find('a')['href'])
        urls.append(url)
    return urls


urls_ = [f'https://www.hse.ru/edu/vkr/?page={i}' for i in range(100, 104)]
parser = AsyncParser(urls_, get_urls_from_page, 10)
results = parser.parse()
print(len(results))
print([len(i) for i in results])
print(results)
