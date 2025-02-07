import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate

from assets.function.which import WHICH
from assets.function.similar import WS1

app = dash.Dash(__name__, suppress_callback_exceptions=True)

server = app.server

# ---------------------------------------------------------
# Styles
# ---------------------------------------------------------
button_style_analysis = {
    "backgroundColor": "#78c5f9",
    "color": "#1d1f21",
    "border": "none",
    "padding": "8px 16px",
    "borderRadius": "4px",
    "cursor": "pointer",
    "fontWeight": "600"
}

button_style_selection = {
    "backgroundColor": "#1d1f21",
    "color": "#c5c8c6",
    "border": "none",
    "borderRadius": "4px",
    "cursor": "pointer",
    "fontWeight": "600",
    "height": "40px",
    "width": "50%",
    "marginBottom": "10px"
}

input_style_selection = {
    "width": "50%",
    "height": "40px",
    "color": "#c5c8c6",
    "backgroundColor": "#1d1f21",
    "border": "1px solid #c5c8c6",
    "borderRadius": "4px",
    "marginBottom": "10px"
}

# ---------------------------------------------------------
# Layout
# ---------------------------------------------------------
app.layout = html.Div(
    style={
        "display": "flex",
        "flexDirection": "column",
        "width": "100%",
        "minHeight": "100vh",
        "margin": "0px",
        "backgroundColor": "#1d1f21"
    },
    children=[
        # Top row: Eye image + H1 side by side
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "width": "100%",
                "padding": "10px"
            },
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "alignItems": "center"
                    },
                    children=[
                        html.H1(
                            "Equity",
                            className="anvilDrop",
                            style={
                                "textAlign": "justify",
                                "fontSize": "5rem",
                                "margin": "0px",
                                "color": "#fff"
                            }
                        ),
                        html.Img(
                            src=app.get_asset_url("eye.png"),
                            style={
                                "width": "100px",
                                "marginLeft": "5px"
                            }
                        ),
                        html.H1(
                            "pportunitiy Gauge",
                            className="anvilDrop",
                            style={
                                "textAlign": "justify",
                                "fontSize": "5rem",
                                "margin": "0px",
                                "color": "#fff"
                            }
                        ),
                    ],
                ),
            ]
        ),

        # Tabs
        dcc.Tabs(
            id="tabs",
            value='tab-1',
            className='dash-tabs',
            children=[
                # -----------------------------------------------------------------
                # TAB 1: Selection
                # -----------------------------------------------------------------
                dcc.Tab(
                    label='Selection',
                    value='tab-1',
                    className="fade-in-section",
                    children=[
                        html.Div(
                            style={
                                "padding": "20px",
                                "display": "flex",
                                "flexDirection": "column",
                                "alignItems": "flex-start"
                            },
                            children=[
                                html.H3("Enter a Symbol to Retrieve Similar Stocks", style={"color": "#fff"}),
                                dcc.Input(
                                    id='symbol-input',
                                    type='text',
                                    placeholder='e.g. AAPL',
                                    value='AAPL',
                                    style=input_style_selection
                                ),
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "width": "100%",
                                        "gap": "10px",
                                        "marginTop": "10px"
                                    },
                                    children=[
                                        html.Button(
                                            "Get Similar Stocks",
                                            id="similar-button",
                                            n_clicks=0,
                                            style={
                                                "backgroundColor": "#1d1f21",
                                                "color": "#c5c8c6",
                                                "border": "none",
                                                "borderRadius": "4px",
                                                "cursor": "pointer",
                                                "fontWeight": "600",
                                                "height": "40px",
                                                # Expand horizontally
                                                "flex": "1"
                                            }
                                        )
                                    ]
                                ),
                                html.Div(
                                    id='similar-output',
                                    style={'marginTop': '20px', "color": "#fff"}
                                ),
                            ]
                        ),
                    ]
                ),

                # -----------------------------------------------------------------
                # TAB 2: Analysis
                # -----------------------------------------------------------------
                dcc.Tab(
                    label='Analysis',
                    value='tab-2',
                    className="fade-in-section",
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "width": "100%",
                                "gap": "20px",
                                "padding": "20px"
                            },
                            children=[
                                # Left Column
                                html.Div(
                                    style={
                                        "flex": "1",
                                        "minWidth": "250px",
                                        "maxWidth": "600px"
                                    },
                                    children=[
                                        html.H3("Enter Tickers Below", style={"color": "#fff"}),
                                        dash_table.DataTable(
                                            id='ticker-table',
                                            columns=[{"name": "Ticker", "id": "ticker"}],
                                            data=[{"ticker": ""} for _ in range(5)],
                                            editable=True,
                                            row_deletable=True,
                                            style_header={"backgroundColor": "#2a2d31", "color": "#fff"},
                                            # Ensure typed text is white
                                            style_cell={
                                                "backgroundColor": "#1d1f21",
                                                "color": "#fff"
                                            }
                                        ),
                                        html.Br(),
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "width": "100%",
                                                "gap": "10px"
                                            },
                                            children=[
                                                html.Button(
                                                    "Add Row",
                                                    id="add-row-button",
                                                    n_clicks=0,
                                                    style={
                                                        "backgroundColor": "white",
                                                        "color": "black",
                                                        "border": "none",
                                                        "padding": "8px 16px",
                                                        "borderRadius": "4px",
                                                        "cursor": "pointer",
                                                        "fontWeight": "600",
                                                        # Expand horizontally
                                                        "flex": "1"
                                                    }
                                                ),
                                                html.Button(
                                                    "Clear Table",
                                                    id="clear-table",
                                                    n_clicks=0,
                                                    style={
                                                        "backgroundColor": "white",
                                                        "color": "black",
                                                        "border": "none",
                                                        "padding": "8px 16px",
                                                        "borderRadius": "4px",
                                                        "cursor": "pointer",
                                                        "fontWeight": "600",
                                                        # Expand horizontally
                                                        "flex": "1"
                                                    }
                                                ),
                                            ]
                                        ),
                                        html.Hr(),
                                    ]
                                ),

                                # Right Column
                                html.Div(
                                    style={
                                        # Make this column a flex container
                                        "flex": "2",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "minWidth": "300px"
                                    },
                                    children=[
                                        # A container that can flex to fill vertical space
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "gap": "20px",
                                                # Let this row expand if there's extra height
                                                "flex": "1",
                                                "alignItems": "stretch",
                                                "marginBottom": "20px"
                                            },
                                            children=[
                                                html.Button(
                                                    "Run Analysis",
                                                    id="run-analysis",
                                                    n_clicks=0,
                                                    style={
                                                        "backgroundColor": "white",
                                                        "color": "black",
                                                        "border": "none",
                                                        "padding": "8px 16px",
                                                        "borderRadius": "4px",
                                                        "cursor": "pointer",
                                                        "fontWeight": "600",
                                                        # Let the button fill the space
                                                        "flex": "1",
                                                        "alignSelf": "stretch"
                                                    }
                                                ),
                                                # Loading spinner
                                                dcc.Loading(
                                                    id="analysis-loading",
                                                    type="dot",
                                                    color="#fff",
                                                    children=[
                                                        # Graph is hidden by default; reveals after analysis
                                                        dcc.Graph(
                                                            id='analysis-graph',
                                                            style={"display": "none"}
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                        # Additional content below the button/spinner row
                                        html.H3(
                                            "Select Tests To Include",
                                            style={'marginTop': '5rem', "color": "#fff"}
                                        ),
                                        dcc.Checklist(
                                            id='test-checklist',
                                            options=[
                                                {'label': "Short Interest %", 'value': 'two'},
                                                {'label': "Income / Employees", 'value': 'three'},
                                                {'label': "Return CV", 'value': 'five'},
                                                {'label': "Insider Ownership", 'value': 'six'},
                                                {'label': "ForwardPE/PE", 'value': 'seven'},
                                                {'label': "Momentum", 'value': 'eight'},
                                                {'label': "Beta", 'value': 'nine'},
                                                {'label': "Growth Score", 'value': 'ten'},
                                                {'label': "ROE", 'value': 'twelve'},
                                                {'label': "ROA", 'value': 'thirteen'},
                                                {'label': "Cash/Price", 'value': 'fourteen'},
                                                {'label': "Analyst Recommendations", 'value': 'fifteen'},
                                                {'label': "Profit Margin", 'value': 'sixteen'},
                                                {'label': "Price / Free Cash Flow", 'value': 'seventeen'},
                                                {'label': "Price /Earnings", 'value': 'eighteen'},
                                                {'label': "Debt/Equity", 'value': 'nineteen'},
                                                {'label': "Gross Margin", 'value': 'twenty'},
                                                {'label': "AvgVolume / SharesFloat", 'value': 'twentyone'},
                                                {'label': "ShortInterest / AvgVolume", 'value': 'twentytwo'},
                                            ],
                                            value=[
                                                'two','three','five','six','seven','eight',
                                                'nine','ten','twelve','thirteen','fourteen',
                                                'fifteen','sixteen','seventeen','eighteen',
                                                'nineteen','twenty','twentyone','twentytwo'
                                            ],
                                            style={'color': '#fff'}
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),

                # -----------------------------------------------------------------
                # TAB 3: Biography
                # -----------------------------------------------------------------
                dcc.Tab(
                    label='Biography',
                    value='tab-3',
                    className="fade-in-section"
                )
            ]
        ),

        # Content that changes based on the selected tab
        html.Div(
            id='tabs-content',
            className="fade-in-section",
            style={"width": "100%", "padding": "20px"}
        )
    ]
)

# ---------------------------------------------------------
# Callbacks
# ---------------------------------------------------------
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def update_tab_content(tab):
    if tab == 'tab-1':
        return html.Div()  # Additional content for Tab 1 if desired

    elif tab == 'tab-2':
        return html.Div()  # Additional content for Tab 2 if desired

    elif tab == 'tab-3':
        # The "Bibliography" (or About) tab
        return html.Div(
            dcc.Markdown(
                """
                ## About the Equity Opportunity Gauge

                The **Equity Opportunity Gauge** is a stock-screening and analysis tool designed to help investors
                quickly evaluate multiple companies and compare their relative strengths across a variety of metrics.
                By focusing on the most pertinent tests from both fundamental and technical perspectives, the app
                offers a convenient visual ranking to highlight which tickers may present attractive opportunities.

                ### How It Works
                1. **Ticker Input**:
                   You begin by entering the stock tickers you want to analyze in the “Analysis” tab. Each ticker
                   is gathered and used in a series of computational “tests.”

                2. **Selected Tests**:
                   Under “Select Tests To Include,” you can decide which metrics or tests are relevant to your
                   evaluation. For example, you may only want to consider fundamentals such as Debt/Equity and
                   Profit Margin, or you may want to include momentum-oriented tests like short interest or beta.

                3. **Ranking Method**:
                   - Each test assigns a score (or pass/fail) based on how well a ticker performs relative to others.
                   - These scores are summed to produce a **total** score for each ticker.
                   - Tickers are displayed in descending order (highest to lowest) in a bar chart.
                   - **Color-Coded Bars** highlight which metrics contribute most to each company's overall rank.

                ### Tests & Metrics
                 1. **Short Interest %**  
                   - Measures the percentage of a company’s float that is held by short sellers.  
                   - A higher short interest can suggest negative market sentiment or potential for a short squeeze.

                2. **Income / Employees**  
                   - Assesses how much net income (or revenue) is generated per employee.  
                   - Higher values imply greater productivity and profitability per staff member.

                3. **Return CV**  
                   - CV stands for Coefficient of Variation, used to measure the volatility relative to returns.  
                   - A lower CV can indicate more stable returns or efficiency in generating returns relative to risk.

                4. **Insider Ownership**  
                   - Tracks the percentage of shares owned by corporate insiders.  
                   - Higher insider ownership can suggest strong confidence from management in the company’s prospects.

                5. **Forward P/E vs. P/E**  
                   - Compares the Forward Price-to-Earnings ratio against the current P/E ratio.  
                   - A significantly lower Forward P/E may indicate expected growth or improved future earnings.

                6. **Momentum**  
                   - Evaluates how quickly a stock’s price is moving, often based on average returns over a certain period.  
                   - Positive momentum can suggest strong short-term investor demand.

                7. **Beta**  
                   - A measure of a stock’s volatility compared to the broader market.  
                   - A beta above 1.0 is more volatile; below 1.0 is less volatile.

                8. **Growth Score**  
                   - A combined metric incorporating revenue/earnings growth rates.  
                   - High-growth companies often have stronger prospects but can carry higher valuations.

                9. **ROE** (Return on Equity)  
                   - Calculated as net income divided by shareholders’ equity.  
                   - Measures how effectively management is using equity to generate profits.

                10. **ROA** (Return on Assets)  
                    - Calculated as net income divided by total assets.  
                    - Assesses how efficiently a company is using its assets to produce earnings.

                11. **Cash/Price**  
                    - Amount of cash per share relative to its share price.  
                    - Higher levels of cash per share can offer stability and flexibility.

                12. **Analyst Recommendations**  
                    - Aggregates analyst buy/hold/sell ratings into an overall score.  
                    - Used as a consensus indicator of market sentiment.

                13. **Profit Margin**  
                    - The ratio of net income to revenue.  
                    - Higher profit margins typically indicate better operational efficiency.

                14. **Price / Free Cash Flow**  
                    - Compares the stock price to its free cash flow per share.  
                    - Often a strong indicator of value; lower values suggest an undervalued situation.

                15. **Price / Earnings**  
                    - Divides share price by earnings per share.  
                    - Provides a quick snapshot of how the market values the company’s earnings.

                16. **Debt/Equity**  
                    - Measures how much debt a company carries compared to shareholder equity.  
                    - Lower ratios generally imply less leverage and potentially lower financial risk.

                17. **Gross Margin**  
                    - Net sales minus cost of goods sold, expressed as a percentage of revenue.  
                    - Higher gross margins mean more resources for R&D, marketing, and other activities.

                18. **AvgVolume / SharesFloat**  
                    - Evaluates daily trading volume relative to the total shares available (float).  
                    - High volume relative to float can mean more liquidity or active trading.

                19. **ShortInterest / AvgVolume**  
                    - Indicates how many days of average trading volume are held in short positions.  
                    - High levels may signal potential for short squeezes or heightened volatility.

                ### Data Source
                All data and fundamental metrics are sourced from **[FinViz](https://finviz.com/)**, ensuring
                that each test uses accurate, up-to-date market information.
                """
            ),
            style={"color": "#fff"}
        )

    return html.Div()

@app.callback(
    Output('ticker-table', 'data'),
    Input('add-row-button', 'n_clicks'),
    Input('clear-table', 'n_clicks'),
    State('ticker-table', 'data'),
    State('ticker-table', 'columns')
)
def update_table(add_clicks, clear_clicks, rows, columns):
    ctx = dash.callback_context
    if not ctx.triggered:
        return rows

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'clear-table':
        return [{"ticker": ""} for _ in range(5)]
    elif triggered_id == 'add-row-button':
        rows.append({c['id']: "" for c in columns})
        return rows
    else:
        return rows

@app.callback(
    Output('analysis-graph', 'figure'),
    Output('analysis-graph', 'style'),
    Input('run-analysis', 'n_clicks'),
    State('ticker-table', 'data'),
    State('test-checklist', 'value')
)
def run_analysis_and_plot(n_clicks, table_data, selected_tests):
    # If button not clicked, don't update anything
    if not n_clicks:
        raise PreventUpdate

    # Clean up ticker input
    tickers = []
    for row in table_data:
        raw_value = row.get('ticker', '')
        ticker_str = str(raw_value).strip().upper()
        if ticker_str:
            tickers.append(ticker_str)

    # If no tickers, return an empty figure
    if not tickers:
        fig = go.Figure()
        fig.update_layout(
            title="No tickers entered.",
            paper_bgcolor="#1d1f21",
            plot_bgcolor="#1d1f21",
            font_color="#fff"
        )
        return fig, {"display": "block", "width": "100%", "height": "500px"}

    # Instantiate the analyzer
    analyzer = WHICH(tickers)
    analyzer.run_tests_selected(selected_tests)

    # Summation for sorting
    analyzer.comparison['total'] = analyzer.comparison[selected_tests].sum(axis=1)
    analyzer.comparison = analyzer.comparison.sort_values(by='total', ascending=False)

    # Build the figure
    fig = analyzer.results_plotly(top_n=len(analyzer.comparison))
    fig.update_layout(
        paper_bgcolor="#1d1f21",
        plot_bgcolor="#1d1f21",
        font_color="#fff"
    )

    return fig, {"display": "block", "width": "100%", "height": "500px"}

@app.callback(
    Output('similar-output', 'children'),
    Input('similar-button', 'n_clicks'),
    State('symbol-input', 'value')
)
def get_similar_stocks(n_clicks, symbol):
    if n_clicks < 1:
        return dash.no_update

    symbol = symbol.strip().upper()
    if not symbol:
        return "Please enter a symbol."

    scraper = WS1(symbol)
    similar_list = scraper.scrape()
    if not similar_list:
        return f"No related symbols found for {symbol}."

    return html.Ul([html.Li(sym) for sym in similar_list])

# ---------------------------------------------------------
# Run the app
# ---------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True, port=8151)
