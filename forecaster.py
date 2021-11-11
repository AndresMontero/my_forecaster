# pip install pandas numpy matplotlib streamlit pystan fbprophet cryptocmd plotly
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, datetime

from cryptocmd import CmcScraper
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import csv
import time


def load_data(selected_ticker):
    init_scraper = CmcScraper(selected_ticker)
    df = init_scraper.get_dataframe()
    min_date = pd.to_datetime(min(df['Date']))
    max_date = pd.to_datetime(max(df['Date']))
    return min_date, max_date


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





possible_crypto = ['BTC', 'BCH', 'DOGE', 'ETH', 'LTC', 'XRP', 'XLM',
                   'EOS', 'OMG', 'ZRX', 'XTZ', 'BNT', 'ADA', 'FIL', 'LRC',
                   'OXT', 'SNX', 'GRT', 'UMA', 'UNI',
                   'YFI', 'ATOM', 'NKN', 'Matic', 'Algo', 'Celo', 'Band',
                   'Link', 'TRB', 'AAVE', 'COMP', 'CRV', 'SUSHI', '1inch',
                   'MIR', 'OGN', 'BAT', 'ENJ', 'SKL', 'CTSI', 'STORJ',
                   'RLC', 'DOT', 'SOL', 'ICP', 'KNC', 'KEEP', 'MANA',
                   'ANKR', 'AMP', 'CHZ']

coins_to_invest = []

days_to_predict = 3

for coin in possible_crypto:
    print(f"\n\n\n\n\n\n\n ***************************************** \n Running Analysis for coin {coin} \n\n\n\n\n")
    min_date, max_date = load_data(coin)
    scraper = CmcScraper(coin)
    data = scraper.get_dataframe()
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    df_train.sort_values(by='ds', key=pd.to_datetime, inplace=True)
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

    ### Predict using the model
    future = m.make_future_dataframe(periods=days_to_predict)
    forecast = m.predict(future)

    ### Evaluate the predictions of the days_to_predict next days
    average_next = forecast['yhat'][-days_to_predict:].mean()
    average_previous = df_train['y'][-days_to_predict:].mean()

    if average_next > (average_previous * 1.2):
        print(f"Adding coin: {coin} to the list of investment")
        coins_to_invest.append(coin)

        ### Show and plot forecast for the coins to invest in
        fig1 = plot_plotly(m, forecast)
        fig1.update_layout(title=f"The coin is {coin}")
        fig1.show()
        # plt.subplots(figsize=(12, 6))
        # m.plot_components(forecast)
        # plt.show(block=False)

print(f"The coins to invest are {coins_to_invest}")
timestr = time.strftime("%Y%m%d-%H%M%S")
with open(f'./predictions/Predictions_{timestr}.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(coins_to_invest)
