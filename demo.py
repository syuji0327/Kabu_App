import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# シミュレーション関数
def simulate_trading(data, start_date, end_date, initial_funds, transaction_fee):
    data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    balance = initial_funds
    stocks = 0
    holdings = []  # 各日の評価額を格納するリスト

    for index, row in data.iterrows():
        decision = np.random.choice(['Buy', 'Sell', 'Hold'])  # ランダムに売買判断

        if decision == 'Buy':
            if balance >= row['Open'] + transaction_fee:  # 手数料を考慮して購入可能な場合
                stocks_to_buy = (balance - transaction_fee) // row['Open']
                stocks += stocks_to_buy
                balance -= (stocks_to_buy * row['Open'] + transaction_fee)

        elif decision == 'Sell':
            if stocks > 0:
                balance += stocks * row['Open']
                balance -= transaction_fee
                stocks = 0

        holdings.append(balance + stocks * row['Close'])

    return holdings

# メイン部分
st.title('株価シミュレーションと評価額グラフ')

uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    
    st.subheader('株価データ')
    st.dataframe(data)

    # サイドバーからシミュレーション設定を取得
    min_date = data['Date'].min()
    max_date = data['Date'].max()
    start_date = st.sidebar.date_input("シミュレーション開始日", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("シミュレーション終了日", max_date, min_value=min_date, max_value=max_date)
    start_date = pd.to_datetime(start_date)  # datetime64[ns]型に変換
    end_date = pd.to_datetime(end_date)      # datetime64[ns]型に変換
    initial_funds = st.sidebar.number_input("初期資金 (円)", min_value=0)
    transaction_fee = 500  # 手数料
    
    holdings = simulate_trading(data, start_date, end_date, initial_funds, transaction_fee)

    # グラフ表示
    fig, ax = plt.subplots()
    ax.plot(data[data['Date'].between(start_date, end_date)]['Date'], holdings, label='評価額')
    ax.set_xlabel('日時')
    ax.set_ylabel('評価額')
    ax.legend()
    st.subheader('評価額のグラフ')
    st.pyplot(fig)
