# pip install pandas numpy matplotlib streamlit pystan fbprophet cryptocmd plotly
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as plt
from datetime import date, datetime

from cryptocmd import CmcScraper
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


def load_data(selected_ticker):
    init_scraper = CmcScraper(selected_ticker)
    df = init_scraper.get_dataframe()
    min_date = pd.to_datetime(min(df['Date']))
    max_date = pd.to_datetime(max(df['Date']))
    return min_date, max_date


# scraper = CmcScraper('ETH')
# data = scraper.get_dataframe()
#
# print(data)
#

### Select ticker & number of days to predict on
selected_ticker = st.sidebar.text_input("Enter coin name to get data (i.e. BTC, ETH, LINK, etc.)", "ETH")


# period = int(st.sidebar.number_input('Number of days to predict:', min_value=0, max_value=1000000, value=365, step=1))
# training_size = int(st.sidebar.number_input('Training set (%) size:', min_value=10, max_value=100, value=100, step=5)) / 100

### Initialise scraper without time interval
@st.cache
def load_data(selected_ticker):
    init_scraper = CmcScraper(selected_ticker)
    df = init_scraper.get_dataframe()
    min_date = pd.to_datetime(min(df['Date']))
    max_date = pd.to_datetime(max(df['Date']))
    return min_date, max_date


data_load_state = st.sidebar.text('Loading data...')
min_date, max_date = load_data(selected_ticker)
data_load_state.text('Loading data... done!')

scraper = CmcScraper(selected_ticker)
data = scraper.get_dataframe()

st.subheader(f'Raw data for coin {selected_ticker}')
st.write(data.head())


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


def plot_raw_data_log():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
    fig.update_yaxes(type="log")
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


# plot_log = st.checkbox("Plot log scale")

plot_raw_data()

make_predictions = st.checkbox("Make predictions")

if make_predictions:

    st.subheader(f'Training Data')

    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    ### Create Prophet model
    m = Prophet(
        changepoint_range=0.85,  # 0.8
        yearly_seasonality='auto',
        weekly_seasonality='auto',
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # multiplicative/additive
        changepoint_prior_scale=0.05
    )

    ### Add (additive) regressor
    for col in df_train.columns:
        if col not in ["ds", "y"]:
            m.add_regressor(col, mode="additive")

    m.fit(df_train)

    st.subheader(f'Predicting Data')

    # 	### Predict using the model
    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)

    ### Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.head())

    st.subheader(f'Forecast plot for {7} days')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.subheader("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
