# File: which.py
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def requests_custom(url):
    """
    Make a GET request with custom headers to mimic a real browser.
    Returns the page content if successful; otherwise None.
    """
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.content
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    return None

def convert_market_cap(market_cap_str):
    """
    Convert finviz-style market cap strings (e.g., '12B', '500M') to float.
    Returns None if unparseable or empty.
    """
    if not market_cap_str:
        return None
    if market_cap_str.endswith('B'):
        return float(market_cap_str[:-1]) * 1e9
    elif market_cap_str.endswith('M'):
        return float(market_cap_str[:-1]) * 1e6
    elif market_cap_str.endswith('K'):
        return float(market_cap_str[:-1]) * 1e3
    else:
        try:
            return float(market_cap_str)
        except ValueError:
            return None

class WHICH:
    """
    A class that:
      1) Downloads price data from yfinance for a list of symbols.
      2) Scrapes Finviz metrics for each symbol.
      3) Builds a comparison DataFrame and runs various tests.
      4) Generates a Plotly stacked bar chart with a dark background.
    """

    def __init__(self, symbols):
        self.symbols = symbols
        self.data = None
        self.metrics_data = {}

        # Initialize comparison DataFrame with float64 dtype
        self.comparison = pd.DataFrame(index=self.symbols).fillna(0).astype("float64")

        # Download data
        self._download_data()
        self._download_metrics()

    # ---------------------------------------------
    # DATA FETCHING
    # ---------------------------------------------
    def _download_data(self):
        """ Download daily price history via yfinance (Close only). """
        symbol_data = {}
        for symbol in self.symbols:
            data = yf.download(tickers=symbol, period='max', interval='1d')
            if not data.empty:
                data = data[['Close']].rename(columns={'Close': symbol})
                symbol_data[symbol] = data
        if symbol_data:
            self.data = pd.concat(symbol_data.values(), axis=1, join='inner')

    def _download_metrics(self):
        """ Fetch Finviz metrics for each symbol. """
        for symbol in self.symbols:
            self.metrics_data[symbol] = self.get_metrics(symbol)  # ✅ Now correctly calling as a method

    def get_metrics(self, ticker):
        """ Scrape key finviz metrics for the given ticker. """
        url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
        html = requests_custom(url)
        if html is None:
            print(f"Failed to retrieve HTML for {ticker}")
            return pd.DataFrame()

        soup = BeautifulSoup(html, 'html.parser')
        metrics_table = soup.find('table', class_='js-snapshot-table snapshot-table2 screener_snapshot-table-body')

        metrics = []
        if metrics_table:
            for row in metrics_table.find_all('tr'):
                cols = row.find_all('td')
                if len(cols) % 2 == 0:
                    for i in range(0, len(cols), 2):
                        metric_name = cols[i].text.strip()
                        metric_value = cols[i + 1].text.strip()
                        metrics.append({'Metric': metric_name, 'Value': metric_value})

        return pd.DataFrame(metrics)

    # ---------------------------------------------
    # RUN ALL TESTS
    # ---------------------------------------------
    def run_tests(self):
        """
        Run all test methods (two, three, five, etc.), then apply weighting.
        """
        # 1) Call each test function
        for test_method in [
            self.two, self.three, self.five, self.six, self.seven,
            self.eight, self.nine, self.ten, self.twelve, self.thirteen,
            self.fourteen, self.fifteen, self.sixteen, self.seventeen,
            self.eighteen, self.nineteen, self.twenty, self.twentyone,
            self.twentytwo
        ]:
            test_method()

        # 2) Convert columns to numeric
        self.comparison = self.comparison.apply(pd.to_numeric, errors='coerce').fillna(0)

        # 3) Weighting Scheme
        weights = {
            'two': 4, 'three': 6, 'five': 5, 'six': 8, 'seven': 5,
            'eight': 4, 'nine': 2, 'ten': 10, 'twelve': 7, 'thirteen': 6,
            'fourteen': 5, 'fifteen': 3, 'sixteen': 8, 'seventeen': 9,
            'eighteen': 9, 'nineteen': 4, 'twenty': 6, 'twentyone': 7,
            'twentytwo': 5
        }
        for col, weight in weights.items():
            if col in self.comparison.columns:
                self.comparison[col] *= weight

    # ---------------------------------------------
    # RUN SELECTED TESTS ONLY
    # ---------------------------------------------
    def run_tests_selected(self, selected_tests, weights=None):
        """
        Run only the test methods specified in 'selected_tests'.
        Then apply weighting (defaults or custom).
        """
        all_tests_map = {
            'two': self.two, 'three': self.three, 'five': self.five,
            'six': self.six, 'seven': self.seven, 'eight': self.eight,
            'nine': self.nine, 'ten': self.ten, 'twelve': self.twelve,
            'thirteen': self.thirteen, 'fourteen': self.fourteen,
            'fifteen': self.fifteen, 'sixteen': self.sixteen,
            'seventeen': self.seventeen, 'eighteen': self.eighteen,
            'nineteen': self.nineteen, 'twenty': self.twenty,
            'twentyone': self.twentyone, 'twentytwo': self.twentytwo
        }
        # 1) Call only the chosen test methods
        for t in selected_tests:
            if t in all_tests_map:
                all_tests_map[t]()

        # 2) Ensure numeric
        self.comparison = self.comparison.apply(pd.to_numeric, errors='coerce').fillna(0)

        # 3) Default or custom weighting
        default_weights = {
            'two': 4, 'three': 6, 'five': 5, 'six': 8, 'seven': 5,
            'eight': 4, 'nine': 2, 'ten': 10, 'twelve': 7, 'thirteen': 6,
            'fourteen': 5, 'fifteen': 3, 'sixteen': 8, 'seventeen': 9,
            'eighteen': 9, 'nineteen': 4, 'twenty': 6, 'twentyone': 7,
            'twentytwo': 5
        }
        if weights is None:
            weights = default_weights

        # 4) Apply weighting for each selected test
        for col in selected_tests:
            if col in self.comparison.columns:
                w = weights.get(col, 1)
                self.comparison[col] *= w

    # ---------------------------------------------
    # RANK HELPERS
    # ---------------------------------------------
    def assign_ranks(self, column):
        """
        Rank values where lower is better.
        ✅ Ensures index alignment between self.comparison and ranked Series.
        """
        if column in self.comparison:
            ranked_series = (
                self.comparison[column]
                .rank(ascending=True, method='min')
                .infer_objects(copy=False)
            )
            
            # ✅ Ensure alignment by reindexing
            self.comparison[column] = ranked_series.reindex(self.comparison.index, fill_value=0)


    def assign_ranks_reverse(self, column):
        """
        Rank values where higher is better.
        ✅ Ensures index alignment between self.comparison and ranked Series.
        """
        if column in self.comparison:
            ranked_series = (
                self.comparison[column]
                .rank(ascending=False, method='min')
                .infer_objects(copy=False)
            )
            
            # ✅ Ensure alignment by reindexing
            self.comparison[column] = ranked_series.reindex(self.comparison.index, fill_value=0)


    def calculate_totals(self):
        """ Compute total ranking scores across all tests. """
        self.comparison = self.comparison.apply(pd.to_numeric, errors='coerce').fillna(0)
        self.comparison['total'] = self.comparison.sum(axis=1)
        return self.comparison['total'].idxmax()

    # ---------------------------------------------
    # TEST METHODS (two, three, five, etc.)
    # ---------------------------------------------
    def two(self):
        """
        Short Interest / Market Cap => rank ascending
        """
        ratios = {}
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty:
                try:
                    short_interest = convert_market_cap(
                        metrics.loc[metrics['Metric'] == 'Short Interest', 'Value'].iloc[0]
                    )
                    market_cap = convert_market_cap(
                        metrics.loc[metrics['Metric'] == 'Market Cap', 'Value'].iloc[0]
                    )
                    if short_interest and market_cap:
                        ratios[symbol] = short_interest / market_cap
                except Exception:
                    ratios[symbol] = None
        self.comparison['two'] = pd.Series(ratios)
        self.assign_ranks('two')

    def three(self):
        """
        Income / Employees => rough measure of profit or revenue efficiency
        """
        scores = {}
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty:
                try:
                    income = convert_market_cap(
                        metrics.loc[metrics['Metric'] == 'Income', 'Value'].iloc[0]
                    )
                    employees = convert_market_cap(
                        metrics.loc[metrics['Metric'] == 'Employees', 'Value'].iloc[0]
                    )
                    # Below are read but not used in the final ratio in this version:
                    beta = float(metrics.loc[metrics['Metric'] == 'Beta', 'Value'].iloc[0])
                    roe = float(metrics.loc[metrics['Metric'] == 'ROE', 'Value'].iloc[0].strip('%')) / 100
                    roa = float(metrics.loc[metrics['Metric'] == 'ROA', 'Value'].iloc[0].strip('%')) / 100
                    if income and employees and beta and roe and roa:
                        scores[symbol] = (income / employees)
                except Exception:
                    scores[symbol] = None
        self.comparison['three'] = pd.Series(scores)
        self.assign_ranks('three')

    def five(self):
        """ Coefficient of Variation of returns => std(returns)/mean(returns) """
        cvs = {}
        if self.data is None:
            return
        for symbol in self.symbols:
            if symbol not in self.data.columns:
                cvs[symbol] = None
                continue
            returns = self.data[symbol].pct_change().dropna()
            if not returns.empty:
                std_dev = returns.std()
                avg_return = returns.mean()

                # FIX: Ensure valid numeric comparison
                if avg_return.notna().all() and not (avg_return == 0).all():
                    cvs[symbol] = std_dev / avg_return
                else:
                    cvs[symbol] = None
            else:
                cvs[symbol] = None

        self.comparison['five'] = pd.Series(cvs).fillna(0)
        self.assign_ranks('five')

    def six(self):
        """
        Insider Ownership => rank ascending
        """
        insider_ownership = {}
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty:
                try:
                    insider_own = float(
                        metrics.loc[metrics['Metric'] == 'Insider Own', 'Value'].iloc[0].strip('%')
                    ) / 100
                    insider_ownership[symbol] = insider_own
                except Exception:
                    insider_ownership[symbol] = None
        self.comparison['six'] = pd.Series(insider_ownership)
        self.assign_ranks('six')

    def seven(self):
        """
        Forward P/E / P/E => ratio of forward valuation to current
        """
        ratios = {}
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty:
                try:
                    forward_pe = float(metrics.loc[metrics['Metric'] == 'Forward P/E', 'Value'].iloc[0])
                    pe = float(metrics.loc[metrics['Metric'] == 'P/E', 'Value'].iloc[0])
                    if pe != 0:
                        ratios[symbol] = forward_pe / pe
                except Exception:
                    ratios[symbol] = None
        self.comparison['seven'] = pd.Series(ratios)
        self.assign_ranks('seven')

    def eight(self):
        """
        Momentum => average %change over 60 days
        """
        momentum_scores = {}
        if self.data is None:
            return
        for symbol in self.symbols:
            if symbol not in self.data.columns:
                momentum_scores[symbol] = None
                continue
            try:
                momentum = self.data[symbol].pct_change(60).mean() * 100
                momentum_scores[symbol] = momentum
            except Exception:
                momentum_scores[symbol] = None
        self.comparison['eight'] = pd.Series(momentum_scores)
        self.assign_ranks('eight')

    def nine(self):
        """
        Beta => higher => rank descending
        """
        beta_scores = {}
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty:
                try:
                    beta = float(metrics.loc[metrics['Metric'] == 'Beta', 'Value'].iloc[0])
                    beta_scores[symbol] = beta
                except Exception:
                    beta_scores[symbol] = None
        self.comparison['nine'] = pd.Series(beta_scores)
        self.assign_ranks_reverse('nine')  # higher => better => descending

    def ten(self):
        """
        Growth Score => Sales Y/Y TTM * (Income / Market Cap)
        """
        growth_scores = {}
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty:
                try:
                    sales_yy = float(
                        metrics.loc[metrics['Metric'] == 'Sales Y/Y TTM', 'Value'].iloc[0].strip('%')
                    ) / 100
                    income = convert_market_cap(
                        metrics.loc[metrics['Metric'] == 'Income', 'Value'].iloc[0]
                    )
                    market_cap = convert_market_cap(
                        metrics.loc[metrics['Metric'] == 'Market Cap', 'Value'].iloc[0]
                    )
                    if income and market_cap:
                        growth_scores[symbol] = sales_yy * (income / market_cap)
                except Exception:
                    growth_scores[symbol] = None
        self.comparison['ten'] = pd.Series(growth_scores)
        self.assign_ranks('ten')

    def twelve(self):
        """
        ROE => rank ascending if lower is "worse"
        """
        roe_scores = {}
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty:
                try:
                    roe = float(
                        metrics.loc[metrics['Metric'] == 'ROE', 'Value'].iloc[0].strip('%')
                    ) / 100
                    roe_scores[symbol] = roe
                except Exception:
                    roe_scores[symbol] = None
        self.comparison['twelve'] = pd.Series(roe_scores)
        self.assign_ranks('twelve')  # higher ROE => better => rank ascending => invert logic if needed

    def thirteen(self):
        """
        ROA => rank ascending
        """
        roa_scores = {}
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty:
                try:
                    roa = float(
                        metrics.loc[metrics['Metric'] == 'ROA', 'Value'].iloc[0].strip('%')
                    ) / 100
                    roa_scores[symbol] = roa
                except Exception:
                    roa_scores[symbol] = None
        self.comparison['thirteen'] = pd.Series(roa_scores)
        self.assign_ranks('thirteen')

    def fourteen(self):
        """
        Cash/sh / CurrentPrice => rank ascending (higher is better)
        """
        cash_to_price_scores = {}
        if self.data is None:
            return
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty and symbol in self.data.columns:
                try:
                    cash_per_share = float(
                        metrics.loc[metrics['Metric'] == 'Cash/sh', 'Value'].iloc[0]
                    )
                    recent_price = self.data[symbol].iloc[-1]
                    if recent_price > 0:
                        cash_to_price_scores[symbol] = cash_per_share / recent_price
                    else:
                        cash_to_price_scores[symbol] = None
                except Exception:
                    cash_to_price_scores[symbol] = None
        self.comparison['fourteen'] = pd.Series(cash_to_price_scores)
        self.assign_ranks('fourteen')

    def fifteen(self):
        """
        Recom => lower => better => rank descending
        """
        recom_scores = {}
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty:
                try:
                    recom = float(metrics.loc[metrics['Metric'] == 'Recom', 'Value'].iloc[0])
                    recom_scores[symbol] = recom
                except Exception:
                    recom_scores[symbol] = None
        self.comparison['fifteen'] = pd.Series(recom_scores)
        self.assign_ranks_reverse('fifteen')  # lower => better

    def sixteen(self):
        """
        Profit Margin => higher => better => rank ascending => invert logic if needed
        """
        pm = {}
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty:
                try:
                    profit_margin = float(
                        metrics.loc[metrics['Metric'] == 'Profit Margin', 'Value'].iloc[0].strip('%')
                    ) / 100
                    pm[symbol] = profit_margin
                except Exception:
                    pm[symbol] = None
        self.comparison['sixteen'] = pd.Series(pm)
        self.assign_ranks('sixteen')  # higher => better => might want assign_ranks_reverse if you consider higher better

    def seventeen(self):
        """
        P/FCF => lower => better => rank descending
        """
        p_fcf_scores = {}
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty:
                try:
                    p_fcf = float(
                        metrics.loc[metrics['Metric'] == 'P/FCF', 'Value'].iloc[0]
                    )
                    p_fcf_scores[symbol] = p_fcf
                except Exception:
                    p_fcf_scores[symbol] = None
        self.comparison['seventeen'] = pd.Series(p_fcf_scores)
        self.assign_ranks_reverse('seventeen')  # lower => better => descending

    def eighteen(self):
        """
        P/E => lower => better => rank descending
        """
        pe_scores = {}
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty:
                try:
                    pe = float(
                        metrics.loc[metrics['Metric'] == 'P/E', 'Value'].iloc[0]
                    )
                    pe_scores[symbol] = pe
                except Exception:
                    pe_scores[symbol] = None
        self.comparison['eighteen'] = pd.Series(pe_scores)
        self.assign_ranks_reverse('eighteen')  # lower => better => descending

    def nineteen(self):
        """
        Debt/Eq => lower => better => rank ascending
        """
        debt_scores = {}
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty:
                try:
                    debt_eq_str = metrics.loc[metrics['Metric'] == 'Debt/Eq', 'Value'].iloc[0]
                    debt_eq = float(debt_eq_str)
                    debt_scores[symbol] = debt_eq
                except Exception:
                    debt_scores[symbol] = None
        self.comparison['nineteen'] = pd.Series(debt_scores)
        self.assign_ranks('nineteen')  # lower => better

    def twenty(self):
        """
        Gross Margin => higher => better => typically rank descending => but here using ascending
        """
        gm_scores = {}
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty:
                try:
                    gm_str = metrics.loc[metrics['Metric'] == 'Gross Margin', 'Value'].iloc[0]
                    gross_margin = float(gm_str.strip('%')) / 100
                    gm_scores[symbol] = gross_margin
                except Exception:
                    gm_scores[symbol] = None
        self.comparison['twenty'] = pd.Series(gm_scores)
        # If you want higher => better => rank descending, do self.assign_ranks_reverse
        # But here, let's keep ascending:
        self.assign_ranks_reverse('twenty')  # higher => better => descending

    def twentyone(self):
        """
        Avg Volume / Shs Float => higher => better => rank descending
        """
        ratios = {}
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty:
                try:
                    avg_volume = convert_market_cap(
                        metrics.loc[metrics['Metric'] == 'Avg Volume', 'Value'].iloc[0]
                    )
                    shs_float = convert_market_cap(
                        metrics.loc[metrics['Metric'] == 'Shs Float', 'Value'].iloc[0]
                    )
                    if avg_volume and shs_float:
                        ratios[symbol] = avg_volume / shs_float
                except Exception:
                    ratios[symbol] = None
        self.comparison['twentyone'] = pd.Series(ratios)
        self.assign_ranks_reverse('twentyone')  # higher => better => descending

    def twentytwo(self):
        """
        Short Interest / Avg Volume => how many days to cover => lower => better => rank ascending
        """
        si_to_av_ratios = {}
        for symbol in self.symbols:
            metrics = self.metrics_data[symbol]
            if not metrics.empty:
                try:
                    short_interest = convert_market_cap(
                        metrics.loc[metrics['Metric'] == 'Short Interest', 'Value'].iloc[0]
                    )
                    avg_volume = convert_market_cap(
                        metrics.loc[metrics['Metric'] == 'Avg Volume', 'Value'].iloc[0]
                    )
                    if short_interest and avg_volume:
                        si_to_av_ratios[symbol] = short_interest / avg_volume
                    else:
                        si_to_av_ratios[symbol] = None
                except Exception as e:
                    print(f"Error processing {symbol} in twentytwo: {e}")
                    si_to_av_ratios[symbol] = None
        self.comparison['twentytwo'] = pd.Series(si_to_av_ratios)
        self.assign_ranks('twentytwo')  # lower => better => ascending

    # ---------------------------------------------
    # PLOTLY RESULTS WITH DARK BACKGROUND
    # ---------------------------------------------
    def results_plotly(self, top_n=50):
        """ Generate a Plotly stacked bar chart of top-ranked stocks. """
        self.calculate_totals()
        top_symbols = self.comparison.nlargest(top_n, 'total')
        tests = [col for col in self.comparison.columns if col != 'total']
        data = top_symbols[tests]

        fig = go.Figure()
        sorted_symbols = data.index.tolist()
        base_vals = [0] * len(sorted_symbols)

        test_descriptions = {'five': "Return CV (Risk)"}

        for test_col in tests:
            y_vals = data[test_col].values
            fig.add_trace(
                go.Bar(
                    x=y_vals,
                    y=sorted_symbols,
                    orientation='h',
                    name=test_descriptions.get(test_col, test_col),
                    base=base_vals
                )
            )
            base_vals = [base_vals[i] + y_vals[i] for i in range(len(base_vals))]

        fig.update_layout(
            barmode='stack',
            title="Top Ranked Symbols",
            yaxis=dict(autorange='reversed'),
            paper_bgcolor="#1d1f21",
            plot_bgcolor="#1d1f21",
            font=dict(color="#c5c8c6"),
            legend=dict(title="Tests"),
            margin=dict(l=120, r=50, t=50, b=50),
        )

        return fig


    # ---------------------------------------------
    # ORIGINAL MATPLOTLIB RESULTS
    # ---------------------------------------------
    def results(self):
        """
        Matplotlib-based stacked bar chart + optional heatmap (original code).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        self.calculate_totals()
        top_n = 50
        top_symbols = self.comparison.nlargest(top_n, 'total')
        tests = [col for col in self.comparison.columns if col != 'total']
        data = top_symbols[tests]

        palette = sns.color_palette("muted", len(tests))
        plt.figure(figsize=(12, 8))
        bottom = np.zeros(len(data))

        test_descriptions = {
            'two': "Short Interest %",
            'three': "Income/Employees (adj)",
            'five': "Return CV (Risk)",
            'six': "Insider Ownership",
            'seven': "Forward PE / PE",
            'eight': "Momentum",
            'nine': "Beta",
            'ten': "Growth Score",
            'twelve': "ROE",
            'thirteen': "ROA",
            'fourteen': "Cash/Price",
            'fifteen': "Recom",
            'sixteen': "Profit Margin",
            'seventeen': "P/FCF",
            'eighteen': "P/E",
            'nineteen': "Debt/Equity",
            'twenty': "Gross Margin",
            'twentyone': "Trading Activity",
            'twentytwo': "Days to Cover Shorts"
        }

        # Plot stacked bars with Seaborn
        for i, test in enumerate(tests):
            label = test_descriptions.get(test, test)
            sns.barplot(
                x=data[test], y=data.index,
                left=bottom, label=label,
                color=palette[i]
            )
            bottom += data[test]

        plt.xlabel('Total Weighted Score', fontsize=12)
        plt.ylabel('Symbols', fontsize=12)
        plt.title(f'Top {len(top_symbols)} Ranked Symbols (Capped at {top_n})', fontsize=16)
        plt.legend(title="Tests", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

        # Example optional correlation heatmap below...
        # (Omitted for brevity; re-add if needed)

# -----------------------------------
# Usage Example
# -----------------------------------
if __name__ == "__main__":
    # Some test symbols
    test_symbols = ["NVDA", "AMZN", "PLTR"]
    x = WHICH(test_symbols)
    x.run_tests()  # Run ALL tests & weighting

    # If you want a Plotly stacked bar in dark theme:
    fig = x.results_plotly(top_n=3)
    fig.show()
