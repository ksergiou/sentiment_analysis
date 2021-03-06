from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

# Web-scraping using BeautifulSoup

# Scrape finviz
finviz_url = 'https://finviz.com/quote.ashx?t='

news_tables = {}

for ticker in tickers:
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)

    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

# List of lists, each containing the headline and date
parsed_data = []

for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.text
        date_data = row.td.text.split(' ')
        if len(date_data) == 1:
            time = date_data[0][0:7]
        else:
            date = date_data[0]
            time = date_data[1][0:7]
        parsed_data.append([ticker, date, time, title])

# Put everything into a df
df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
df.to_csv('news_df.csv',index=False)