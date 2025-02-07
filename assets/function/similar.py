{\rtf1\ansi\ansicpg1252\cocoartf2759
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import requests\
from bs4 import BeautifulSoup\
\
class WS1:\
    def __init__(self, ticker):\
        """\
        Initialize with a ticker symbol. Converts ticker to uppercase\
        and constructs the FinViz URL.\
        """\
        self.ticker = str(ticker).upper()\
        self.url = f"https://finviz.com/quote.ashx?t=\{self.ticker\}&ty=c&ta=1&p=d"\
\
    def scrape(self):\
        """\
        Scrapes FinViz for the provided ticker, returning a list of related symbols\
        from the <span style="font-size:11px"> element (all <a> with class='tab-link').\
        """\
        headers = \{\
            "User-Agent": (\
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "\
                "AppleWebKit/537.36 (KHTML, like Gecko) "\
                "Chrome/106.0.0.0 Safari/537.36"\
            )\
        \}\
        try:\
            resp = requests.get(self.url, headers=headers)\
            resp.raise_for_status()\
        except Exception as e:\
            print(f"Error requesting FinViz page: \{e\}")\
            return []\
\
        soup = BeautifulSoup(resp.text, "html.parser")\
\
        span = soup.find("span", \{"style": "font-size:11px"\})\
        if not span:\
            return []\
\
        # Within this span, find all <a class="tab-link">\
        links = span.find_all("a", class_="tab-link")\
        if not links:\
            return []\
\
        # Extract the text (symbol) from each link\
        related_symbols = [link.get_text(strip=True) for link in links]\
        return related_symbols\
}