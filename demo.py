import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# シミュレーション関数
def simulate_random_trading(data, start_date, end_date, initial_funds, transaction_fee, min_purchase_unit):
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


# シミュレーション関数（予測売買判断）
def simulate_predicted_trading(data, start_date, end_date, initial_funds, transaction_fee, min_purchase_unit):
    data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    balance = initial_funds
    stocks = 0
    holdings = []

    for index, row in data.iterrows():
        if index + 1 < len(data):
            next_day_open = data.at[index + 1, 'Open']
            next_day_close = data.at[index + 1, 'Close']

            if next_day_close > row['Close'] + transaction_fee:
                # パターンA
                if balance >= next_day_open + transaction_fee:
                    stocks_to_buy = (balance - transaction_fee) // next_day_open // min_purchase_unit * min_purchase_unit
                    stocks += stocks_to_buy
                    balance -= (stocks_to_buy * next_day_open + transaction_fee)
            elif next_day_close < row['Close'] - transaction_fee:
                # パターンB
                if stocks > 0:
                    balance += stocks * next_day_open
                    balance -= transaction_fee
                    stocks = 0
            else:
                # パターンC
                pass

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
    min_purchase_unit = st.sidebar.radio("最小購入単位", options=[1, 100], index=1)  # 100株を選択

    holdings_random = simulate_random_trading(data, start_date, end_date, initial_funds, transaction_fee, min_purchase_unit)
    holdings_predicted = simulate_predicted_trading(data, start_date, end_date, initial_funds, transaction_fee, min_purchase_unit)
    final_evaluation_random = holdings_random[-1]
    final_evaluation_predicted = holdings_predicted[-1]

    # グラフ表示
    fig1, ax1 = plt.subplots()
    ax1.plot(data[data['Date'].between(start_date, end_date)]['Date'], holdings_random, label='ランダム売買')
    ax1.set_xlabel('日時')
    ax1.set_ylabel('評価額')
    ax1.legend()
    ax1.text(end_date, final_evaluation_random, f'最終評価額: {final_evaluation_random:.0f}', verticalalignment='bottom', horizontalalignment='right', fontsize=10)
    st.subheader('ランダム売買の評価額のグラフ')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(data[data['Date'].between(start_date, end_date)]['Date'], holdings_predicted, label='予測売買')
    ax2.set_xlabel('日時')
    ax2.set_ylabel('評価額')
    ax2.legend()
    ax2.text(end_date, final_evaluation_predicted, f'最終評価額: {final_evaluation_predicted:.0f}', verticalalignment='bottom', horizontalalignment='right', fontsize=10)
    st.subheader('予測売買の評価額のグラフ')
    st.pyplot(fig2)