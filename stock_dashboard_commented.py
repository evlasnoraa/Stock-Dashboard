"""
Stock Dashboard: A web app that is inspired by popular stock trading apps like Freetrade, 
Rise by Motilal Oswaal, and ofcourse, Yahoo Finance. 

Features:
- Create dynamic dashboard for any company available on Yahoo Finance.
- Display a stock price chart for different time periods, that can be selected.
- Display summary statisitcs for different time periods that capture the stock performance.
- Display a 30-day forecast using the Holt-Winters Triple Exponential Smoothing algorithm.

Requirements: Streamlit, plotly, pandas, yfinance API, statsmodel, time

Run instructions: 
1) Make sure you are in the same directory as your .py file. 
2) Use pip install command to install all libraries. Use pip3 install if pip install doesn't work. 
3) Type this into terminal - streamlit run stock_dashboard.py
4) Enjoy!
"""


import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
import pytz
import ta
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import re
import time

################################################
## Part 1 - Important functions to fetch data ##
################################################

"""
This function calculate the values used in the dynamic portion of the website that updates with changing period/interval.

Args:
ticker   - The short-form name of the stock. For eg. Apple Inc. -> AAPL
period   - The time period over which dashboard needs to be displayed. Ranging from 1 day to max (since the company's inception)
interval - The timestep between each datapoint. Ranges from 1 minute to 1 day. 

Returns:
change       - The change in closing price of the stock between the start and end of the selected time period.
per_change   - The value for change given as a percentage of starting closing price.
high         - The highest value of stock price in the whole period.
low          - The lowest value of stock price in the whole period. 
volume       - Total volume traded over the entire time period. 
last_close   - The closing price at the last datapoint within the time period.
company_data - The entire pandas dataframe containing all the data
"""
def dynamic_stock_data(ticker, period, interval):
    company_data = yf.download(tickers = ticker, period = period, interval = interval)
    # Strip to only one ticker's data, avoiding hierarchial column names
    company_data = company_data.xs(ticker, axis=1, level=1)
    # Returning 'date' as a feature value
    company_data.reset_index(inplace=True)
    # Consistent naming for the Date column
    if 'Datetime' in company_data.columns:
        company_data.rename(columns={'Datetime': 'Date'}, inplace=True)
    # Values for the first datapoint (Starting point)
    first_val = company_data.loc[0]
    # Values for the last datapoint (End point)
    most_recent_vals = company_data.loc[len(company_data)-1]
    # Calculating our return values
    last_close = most_recent_vals.loc['Close']
    change = most_recent_vals.iloc[4] - first_val[4]
    per_change = change/first_val[4]
    high = company_data['High'].max()
    low = company_data['Low'].min()
    volume = company_data['Volume'].sum()
    return change, per_change, high, low, volume, last_close, company_data

# A simple dict to map intervals to periods. 
interval_mapping = {'1d':'1m', '1wk': '1m', '1mo':'2m', '3mo':'1h', '6mo':'4h', '1y':'4h', 'ytd':'4h', '5y':'1d', 'max':'1d'}
# Similar mapping to map periods to strings that will be displayed on graphs for easier reading experience. 
display_mapping = {'1d':'last 1 day', '1wk': 'last 1 week', '1mo':'last 1 month', '3mo':'last 3 months', '6mo':'last 6 months', '1y':'last 1 year', 'ytd':'this year', '5y':'last 5 years', 'max':'its lifetime'}

"""
Plots the data for the above dynamic data that is generated for different time periods/intervals. 

Args:
df_in - Inpute pandas dataframe. It will be the company_data returned by the previous function. 
ticker - The short-form name of the stock. For eg. Apple Inc. -> AAPL
period - The time period over which dashboard needs to be displayed. Ranging from 1 day to max (since the company's inception)
change - The change value returned from the previous function. 

Returns:
fig - A plotly express object that stores info before being plotted later on. 
"""
def plotly_graph(df_in, ticker, period, change):
    df = df_in.copy()
    # Add a column for shortened time labels, avoids clutter
    df['LabelTime'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M')
    # Decide color based on the change values. Positive -> green, Negative -> red. 
    clr = 'red' if change < 0 else 'green'
    fig = px.line(df, x=df.index, y='Close', custom_data = ["LabelTime", "Close"], color_discrete_sequence = [clr])
    # Evenly spaced x-tick values, no matter the period/interval
    tick_step = int(len(df)/10)
    fig.update_xaxes(tickmode='array',tickvals=list(range(0, len(df), tick_step)),
                    ticktext=df['LabelTime'].iloc[::tick_step], tickangle = 45)
    # Adjust all labels to display customized information
    fig.update_layout(xaxis_title="Time (Only Trading Hours)", yaxis_title="Close Price",
                    title={
                    'text': f'{ticker} stock price over {display_mapping[period]}',
                    'x': 0.5,
                    'xanchor': 'center'},
                    margin=dict(t=50, b=50, l=25, r=25))
    # Adjust hover values to display relevant information
    fig.update_traces(hovertemplate="<b>Price:<b> %{customdata[1]}<br>"
                                    "<b>Time:</b> %{customdata[0]}<br>")
    return fig

"""
A combined version of previous two functions dedicated for forecasting values. However, unlike the previous 
functions, this function is not dynamic. It calculates and displays forecasted values for a static 3 year 
time period. 

Args:
ticker - The short-form name of the stock. For eg. Apple Inc. -> AAPL

Returns:
fig - A plotly express object that stores info before being plotted later on.
"""
def forecast_plot(ticker):
    df = yf.download(ticker, period='3y', interval='1d')
    # Strip to only one ticker's data, avoiding hierarchial column names
    df = df.xs(ticker, axis=1, level=1)
    # Returning 'date' as a feature value
    df.reset_index(inplace=True)
    ts = df[['Date', 'Close']].copy()
    ts.set_index('Date', inplace=True)
    # Set business-day frequency and forward fill missing values
    ts = ts.asfreq('B')
    ts['Close'] = ts['Close'].fillna(method='ffill')
    # Fit Holt-Winters model on entire 3 years
    model = ExponentialSmoothing(
        ts['Close'],
        trend='add',
        seasonal='add',
        seasonal_periods=252 # No. of business days in a year
    ).fit()
    # Forecast next 30 trading days
    forecast_steps = 30
    # Starting point for forecast dates is the end point of our current data
    last_date = ts.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='B')
    forecast = model.forecast(forecast_steps)
    forecast.index = forecast_index
    # Create a simple dataframe that can be plotted easily
    values = pd.DataFrame(forecast)
    values.reset_index(inplace=True)
    values.rename(columns = {0:'Close', 'index':'Date'}, inplace=True)
    # Create an 'indicator' col that can be later used for coloring
    values["indicator"] = 'Forecast'
    ts.reset_index(inplace=True)
    ts["indicator"] = 'Actual'
    finale = pd.concat([ts, values])
    # Plot a simple plotly graph with color distinguished using the 'indicator' column
    fig = px.line(finale, x='Date', y='Close', color = 'indicator', color_discrete_sequence = ["#72B7B2", "#FECB52"])
    # Custom labels for axes and graph
    fig.update_layout(xaxis_title="Timeline", yaxis_title="Close Price",
                    title={
                    'text': 'Forecasted Values for the next 30 days',
                    'x': 0.5,
                    'xanchor': 'center'},
                    margin=dict(t=50, b=50, l=25, r=25))
    return fig

"""
This function calculates the summary statistics for the chosen company. Just like function_plot, 
it is a static function and doesn't update with changing periods/intervals.

Args:
ticker - The short-form name of the stock. For eg. Apple Inc. -> AAPL

Returns:
results   - A dictionary containing all the key summary stats that I need. Specifically, it includes 
            the percent change in stock price for different ranges of period, ranging from 1 day to "max". 
            It also contains the EPS and P/E ratio - two keys values looked at by amateur investors. 
sector    - The 'sector' that the company belongs to. Displayed at the beginning of our dashboard.
shortName - The official shortened name of the company. Displayed at the beginning of our dashboard.  
"""
def stats(ticker):
    df = yf.download(ticker, period='1y', interval='1d')
    df = df.xs(ticker, axis=1, level=1)
    df.reset_index(inplace=True)
    #Rename column just in case
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
    ts = df[['Date', 'Close']].copy()
    ts.set_index('Date', inplace=True)
    # First, we get the live price and the latest EPS and P/E ratio
    Ticker = yf.Ticker(ticker).info
    current_price = Ticker.get("regularMarketPrice")
    eps = Ticker.get('trailingEps')
    p_e = Ticker.get('trailingPE')
    sector = Ticker.get('sector')
    shortName = Ticker.get('shortName')
    # Next, we calculate the change in price over the following intervals
    today = date.today()
    results = {}
    periods = {
        'YTD': {'months': today.month - 1, 'days': today.day - 1},
        '1wk': {'days': 7},
        '1mo': {'months': 1},
        '3mo': {'months': 3},
        '6mo': {'months': 6},
        '1yr': {'years': 1}
    }
    for label, delta in periods.items():
        target_date = today - pd.DateOffset(**delta)
        past_data = df[df['Date'] <= target_date]
        if not past_data.empty:
            past_price = past_data.iloc[-1]['Close']
            pct_change = ((current_price - past_price) / past_price) * 100
            results[label] = pct_change
        else:
            results[label] = None
    results['EPS'] = eps
    results['P/E Ratio'] = p_e
    return results, sector, shortName

"""
I wanted our summary stats to be displayed in neat evenly spaced columns and displayed using a custom metric
function rather than streamlit's metric widget that doesn't let you adjust the size of your delta. This function 
achieves both targets. 

Args:
results - The results from above stats function. 

Returns:
N/A (function plots the figure)
"""
def custom_metric(results):
    boxes_html = '''<div style="display: flex; flex-wrap: nowrap; gap: 15px; 
                border: 2px solid #6c757d; border-radius: 10px; font-family: inherit;
                overflow-x: auto; padding: 10px; background-color:rgba(108, 117, 125, 0.1)">
    '''

    items = list(results.items())
    for i, (key, value) in enumerate(items):
        if value is not None:
            color = 'green' if value >= 0 else 'red'
            # Metric box
            boxes_html += f"""<div style="flex: 1; min-width: 100px; display: flex; flex-direction: column; justify-content: center; text-align: center;">
                    <div style="font-size: 20px; font-weight: 500;">{key}</div>
                    <div style="font-size: 20px; font-weight: 400; color: {color};">{value:.2f}%</div>
                </div>
            """
        else: # Handing missing values
            boxes_html += f"""<div style="flex: 1; min-width: 100px; display: flex; flex-direction: column; justify-content: center; text-align: center;">
                    <div style="font-size: 20px; font-weight: 500;">{key}</div>
                    <div style="font-size: 20px; font-weight: 400; color: {"grey"};">{"--"}</div>
                </div>
            """
        # Vertical divider (adaptive height)
        if i < len(items) - 1:
            boxes_html += '''<div style="width: 1px; background-color: #6c757d; height: auto;"></div>
            '''

    boxes_html += '</div>'
    st.markdown(boxes_html, unsafe_allow_html=True)



######################################################
## Part 2 - Time to work on the Streamlit dashboard ##
######################################################

st.set_page_config(layout = 'wide')  
st.title('Real Time Stock Dashboard with Forecast')

# Set up session state variables - these help in retaining key information
if 'submitted' not in st.session_state:
    st.session_state.submitted = False # Remembers company selection was submitted by user
if 'selected_time_period' not in st.session_state:
    st.session_state.selected_time_period = None # Remembers whether a time period was selected by user

# Forecast, stats, sector and name remembers the values for our static dashboard that stays constant
if 'forecast' not in st.session_state:
    st.session_state.forecast= None
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'sector' not in st.session_state:
    st.session_state.sector = None
if 'name' not in st.session_state:
    st.session_state.name = None

# Sidebar setup - Used for selecting company
option = st.sidebar.selectbox('Stock Options', ['Apple(AAPL)', 'Tesla(TSLA)', 'Nvidia(NVDA)', 'GameStop(GME)', 'American Airlines Group(AAL)', 'Something Else'])
# If "Something Else" is seleted, user needs to input the ticker value directly
if option == 'Something Else':
    temp_ticker = st.sidebar.text_input('Please make sure your enter company code!')
else:
    temp_ticker = re.findall(r"\((.*?)\)", option)[0]

# Error handling for unrecognizable tickers or companies with insufficient data
if st.sidebar.button("Submit"):
    st.session_state.submitted = True
    st.session_state.selected_time_period = None  # Reset time period on new submit
    try:
        # We run the forecast_plot and stats functions as soon as user clicks submit, since they static
        st.session_state.final_ticker = temp_ticker.upper()  # normalize
        # These may raise KeyError or ValueError
        st.session_state.forecast = forecast_plot(temp_ticker)
        st.session_state.stats, st.session_state.sector, st.session_state.name = stats(temp_ticker)
        st.sidebar.write("For the best experience, shut the sidebar!")
    except KeyError:
        st.session_state.submitted = False  # prevent further dashboard display
        st.info("Couldn't find the company. Check the company name or hit submit again.")
        st.info("Note: Sometimes the data doesn't load because of inconsistent performance from the yfinance API. Please try re-loading and it should work.")
    except ValueError:
        st.session_state.submitted = False
        st.info(f"Not sufficient data for {temp_ticker}. Need company with more than 1 year of data.")

# Display dashboards after submit
if st.session_state.submitted:
    st.write(" ")
    st.divider()
    c1, c2 = st.columns([5, 5], vertical_alignment = 'center')
    c1.metric("Name", f"{st.session_state.name}")
    c2.metric("Sector", f"{st.session_state.sector}")
    # Dashboard 1 Container
    with st.container(border=True):
        st.write(" ")
        st.markdown("### **Dashboard**")
        time_period = st.segmented_control("abracadabra", options=['1d', '1wk', '1mo', '3mo', '6mo', '1y', 'ytd', '5y', 'max'], selection_mode='single', label_visibility='collapsed')
        st.divider()
        if time_period:
            st.session_state.selected_time_period = time_period
        # Conditionally render only if a time period is selected
        if st.session_state.selected_time_period:
            change, per_change, high, low, volume, last_close, company_data = dynamic_stock_data(st.session_state.final_ticker, time_period, interval_mapping[time_period])
            col_1, col_2, col_3, col_4 = st.columns(4)
            col_1.metric(label=f"{st.session_state.final_ticker} Last Price", value=f"${last_close:.2f}", delta=f"{change:.2f} ({per_change:.2f}%)")
            col_2.metric("High", f"${high:.2f}")
            col_3.metric("Low", f"${low:.2f}")
            col_4.metric("Volume", f"${volume}")
            fig = plotly_graph(company_data, st.session_state.final_ticker, time_period, change)
            st.plotly_chart(fig, use_container_width=True)
        else: # If time period not selected, prompt user to select one.
            st.info("Please select a time period to view the Stock Dashboard.")

    # Dashboard 2 (Static after submit)
    st.write("###")
    st.markdown("### **Summary Stats**")
    st.write("")
    custom_metric(st.session_state.stats)
    st.write("###")
    st.markdown("### **Forecast**")
    st.write("")
    st.plotly_chart(st.session_state.forecast)
    
    
#### Thank you :) ####



 