#import math
import numpy as np
import toml
#import tqdm
# import torch
#import torch.nn as nn
import gspread
import json
from google.oauth2 import service_account
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
# import optuna.integration.lightgbm as lgb_optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#import joblib
#from sklearn import metrics
import streamlit as st


SHEET_NAME = 'sony_2000_2023'

scopes = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive'
]

# デプロイ時有効にする
ACCESS_KEY_JSON = st.secrets["gcp_service_account"]
SPREAD_SHEET_KEY = st.secrets["SpreadSheetKey"]["SPREAD_SHEET_KEY"]
credentials = service_account.Credentials.from_service_account_info( st.secrets["gcp_service_account"], scopes=scopes)

#テスト時有効にする
#ACCESS_KEY_JSON = "env\API_key\kabuapp-d47e1f69fa4b.json"
#SPREAD_SHEET_KEY = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
#credentials = ServiceAccountCredentials.from_json_keyfile_name(ACCESS_KEY_JSON, scopes)

#OAuth2のクレデンシャルを使用してGoogleAPIにログイン
gc = gspread.authorize(credentials)

# IDを指定して、Googleスプレッドシートのワークブックを選択する
workbook = gc.open_by_key(SPREAD_SHEET_KEY)

# シート名を指定して、ワークシートを選択
worksheet = workbook.worksheet(SHEET_NAME)

# スプレッドシートをDataFrameに取り込む
df = pd.DataFrame(worksheet.get_all_values()[1:], columns=worksheet.get_all_values()[0])

#str型から変換
df["Date"] = pd.to_datetime(df["Date"])
df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

# 日時列をDatetime型に変換
df['Date'] = pd.to_datetime(df['Date'])

#df[['Open', 'High', 'Low', 'Close']].plot()

# グラフを作成する関数
def plot_chart(dataframe):
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe['Date'], dataframe['Close'])
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.title('Stock price trends')
    plt.tight_layout()
    st.pyplot()

# 比較グラフを作成する関数
def plot_chart2(dataframe):
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe['Date'], dataframe['Close'],label='Close')
    plt.plot(dataframe['Date'], dataframe['Close_pred'], label='Close_pred')
    plt.plot(dataframe['Date'], dataframe['Open'], label='Open')
    plt.legend()  # 凡例を表示
    plt.xlabel('Date')
    plt.ylabel('Stock price')
    plt.title('Stock price trends')
    plt.tight_layout()
    st.pyplot()


# テクニカル指標を追加する関数
def add_technical_indicators(df):
    # 移動平均を計算して新たな列として追加
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()

    # ボリンジャーバンドを計算して新たな列として追加
    df['SMA'] = df['Close'].rolling(window=20).mean()
    df['std'] = df['Close'].rolling(window=20).std()
    df['upper_band'] = df['SMA'] + 2 * df['std']
    df['lower_band'] = df['SMA'] - 2 * df['std']

    # RSIを計算して新たな列として追加
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rsi = 100 - (100 / (1 + avg_gain / avg_loss))
    df['RSI'] = rsi

    # モメンタムを計算して新たな列として追加
    df['Momentum'] = df['Close'].diff()

    # MACDを計算して新たな列として追加
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']

    # 移動平均乖離率を計算して新たな列として追加
    df['MAP'] = (df['Close'] - df['SMA']) / df['SMA']

    return df.dropna()

# シミュレーション関数　(ランダム売買判断)
def simulate_random_trading(data, start_date, end_date, initial_funds, transaction_fee, min_purchase_unit, trading_standard_fee):
    data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    balance = initial_funds
    stocks = 0
    holdings = []  # 各日の評価額を格納するリスト

    for index, row in data.iterrows():
        decision = np.random.choice(['Buy', 'Sell', 'Hold'])  # ランダムに売買判断

        if decision == 'Buy':
            if balance >= (row['Open'] * min_purchase_unit) + transaction_fee:  # 手数料を考慮して購入可能な場合
                stocks_to_buy = (balance - transaction_fee) // (row['Open'] * min_purchase_unit) * min_purchase_unit
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
def simulate_predicted_trading(data, start_date, end_date, initial_funds, transaction_fee, min_purchase_unit, trading_standard_fee):
    data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    balance = initial_funds
    stocks = 0
    holdings = []

    for index, row in data.iterrows():
        if index + 1 < len(data):
            next_day_open = data.at[index + 1, 'Open']
            next_day_close = data.at[index + 1, 'Close_pred']

            if next_day_close > row['Close'] + trading_standard_fee:
                # パターンA
                print("パターンA(購入)")
                if balance >= (next_day_open * min_purchase_unit) + transaction_fee:
                    stocks_to_buy = (balance - transaction_fee) // (next_day_open * min_purchase_unit) * min_purchase_unit
                    stocks += stocks_to_buy
                    balance -= (stocks_to_buy * next_day_open + transaction_fee)
                    print("購入成功")
                    print("購入株数", stocks_to_buy)
                    print("所持株数", stocks)
                    print("収支", balance)
            elif next_day_close < row['Close'] - trading_standard_fee:
                # パターンB
                print("パターンB(売却)")
                if stocks > 0:
                    balance += stocks * next_day_open
                    balance -= transaction_fee
                    stocks = 0
                    print("所持株数", stocks)
                    print("収支", balance)
            else:
                # パターンC
                print("パターンC(保留)")
                pass

        holdings.append(balance + stocks * row['Close'])

    return holdings

def simulate_trading_with_details(data, start_date, end_date, initial_funds, transaction_fee, min_purchase_unit, trading_standard_fee):
    data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    balance = initial_funds
    stocks = 0
    holdings = []  # 各日の評価額を格納するリスト
    trading_details = []  # 取引詳細を格納するリスト

    for index, row in data.iterrows():
        if index + 1 < len(data):
            next_day_open = data.at[index + 1, 'Open']
            next_day_close = data.at[index + 1, 'Close_pred']
            trade_result = None

            if next_day_close > row['Close'] + trading_standard_fee:
                if balance >= (next_day_open * min_purchase_unit) + transaction_fee:
                    stocks_to_buy = (balance - transaction_fee) // (next_day_open * min_purchase_unit) * min_purchase_unit
                    stocks += stocks_to_buy
                    balance -= (stocks_to_buy * next_day_open + transaction_fee)
                    trade_result = "Buy"
            elif next_day_close < row['Close'] - trading_standard_fee:
                if stocks > 0:
                    balance += stocks * next_day_open
                    balance -= transaction_fee
                    stocks = 0
                    trade_result = "Sell"
            else:
                trade_result = "Hold"

            trading_details.append({
                'index': index,
                'Date': row['Date'],
                'TradeResult': trade_result,
                'StocksToBuy': stocks_to_buy if trade_result == "Buy" else None,
                'Stocks': stocks,
                'Balance': balance,
            })

        holdings.append(balance + stocks * row['Close'])

    # 取引詳細をDataFrameに変換
    trading_details_df = pd.DataFrame(trading_details)

    return trading_details_df


# 新たなテクニカル指標を追加
df2 = add_technical_indicators(df)

df2 = df2.iloc[30:, :]

df2.head(100)

# 学習データとテストデータに分割
train_data = df2[df2["Date"] < "2022-01-01"]  # 9年分のデータ
test_data = df2[("2022-12-31" >= df2["Date"]) & (df2["Date"] >= "2022-01-01")]   # 1年分のデータ

# 特徴量とターゲットに分割
X_train = train_data.drop(['Date', 'Close'], axis=1)
y_train = train_data['Close']

X_test = test_data.drop(['Date', 'Close'], axis=1)
y_test = test_data['Close']

# LightGBM用のデータセットに変換
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)


# LightGBMのパラメータ設定
params = {
    'objective': 'regression',
    'metric': 'mse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 100
}

verbose_eval = 0  # この数字を1にすると学習時のスコア推移がコマンドライン表示される
# early_stoppingを指定してLightGBM学習

# モデルの学習
model = lgb.train(params, 
                  train_data, 
                  valid_sets=[train_data, test_data], 
                  callbacks=[lgb.early_stopping(stopping_rounds=10, 
                                                verbose=True), # early_stopping用コールバック関数
                                                lgb.log_evaluation(verbose_eval)] # コマンドライン出力用コールバック関数
                )

# モデルの保存
model.save_model('stock_price_prediction_model.txt')

# テストデータの予測
test_pred = model.predict(X_test)

# テストデータの予測結果と実際の値との平均二乗誤差を計算
mse = mean_squared_error(y_test, test_pred)
print(f"テストデータの平均二乗誤差：{mse}")

# 保存した学習モデルの読み込み
loaded_model = lgb.Booster(model_file='stock_price_prediction_model.txt')

# 直近1年のテストデータから翌日の株価を推論
recent_data = df2[-365:].drop(columns=['Date', 'Close'])
recent_pred = loaded_model.predict(recent_data)

sim_start_date = "2023-01-01"
sim_end_date = "2023-07-15"

sim_data = df2[(sim_end_date >= df2["Date"]) & (df2["Date"] >= sim_start_date)]   # 1年分のデータ
# 直近1年のテストデータから翌日の株価を推論
sim_data_X = sim_data.drop(columns=['Date', 'Close'])
sim_pred = loaded_model.predict(sim_data_X)

# 推論結果をDataFrameに追加
sim_data['Close_pred'] = sim_pred

sim_data = sim_data.reset_index()

# メイン部分
st.title('株価シミュレーションと評価額グラフ')

uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=['csv'])

# 警告を非表示
st.set_option('deprecation.showPyplotGlobalUse', False)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    
    st.subheader('アップロードした株価データ')
    #元データの表
    st.dataframe(data)
    #元データのグラフ
    plot_chart(df)

   # サイドバーからシミュレーション設定を取得
    min_date = sim_data['Date'].min()
    max_date = data['Date'].max()
    start_date = st.sidebar.date_input("シミュレーション開始日", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("シミュレーション終了日", max_date, min_value=min_date, max_value=max_date)
    start_date = pd.to_datetime(start_date)  # datetime64[ns]型に変換
    end_date = pd.to_datetime(end_date)      # datetime64[ns]型に変換
    initial_funds = st.sidebar.number_input("初期資金 (円)", min_value=0)
    transaction_fee = st.sidebar.number_input("売買手数料 (円)", min_value=0)
    trading_standard_fee = st.sidebar.number_input("許容値動き (円)", min_value=0)
    min_purchase_unit = st.sidebar.radio("最小購入単位", options=[1, 100], index=1)  # 100株を選択

    holdings_random = simulate_random_trading(data, start_date, end_date, initial_funds, transaction_fee, min_purchase_unit, trading_standard_fee)
    holdings_predicted = simulate_predicted_trading(sim_data, start_date, end_date, initial_funds, transaction_fee, min_purchase_unit, trading_standard_fee)
    # デバッグ用
    ditele_df = simulate_trading_with_details(sim_data, start_date, end_date, initial_funds, transaction_fee, min_purchase_unit, trading_standard_fee)

    final_evaluation_random = holdings_random[-1]
    final_evaluation_predicted = holdings_predicted[-1]

    # グラフ表示
    fig1, ax1 = plt.subplots()
    ax1.plot(data[data['Date'].between(start_date, end_date)]['Date'], holdings_random, label='ランダム売買')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('assets [yen]')
    #ax1.text(end_date, final_evaluation_random, f'最終評価額: {final_evaluation_random:.0f}', verticalalignment='bottom', horizontalalignment='right', fontsize=10)
    st.subheader('ランダム売買の評価額のグラフ')
    # シミュレーション結果の最終評価額を表示
    st.write(f"もしあなたが毎日あてずっぽうで売買を行った場合・・・")
    st.write(f"シミュレーション終了日の評価額は{holdings_random[-1]:,.0f} 円です")
    st.write(f"初期資金からの収支は{holdings_random[-1] - initial_funds:,.0f} 円です")
    st.pyplot(fig1)
    

    fig2, ax2 = plt.subplots()
    ax2.plot(sim_data[sim_data['Date'].between(start_date, end_date)]['Date'], holdings_predicted, label='予測売買')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('assets [yen]')
    #ax2.text(end_date, final_evaluation_predicted, f'最終評価額: {final_evaluation_predicted:.0f}', verticalalignment='bottom', horizontalalignment='right', fontsize=10)
    st.subheader('予測売買の評価額のグラフ')
    # シミュレーション結果の最終評価額を表示
    st.write(f"もしあなたが私の予報に従って売買を行った場合・・・")
    st.write(f"シミュレーション終了日の評価額は{holdings_predicted[-1]:,.0f} 円です")
    st.write(f"初期資金からの収支は{holdings_predicted[-1] - initial_funds:,.0f} 円です")
    st.pyplot(fig2)

    #元データの表
    st.write(f"各日の売買履歴")
    st.dataframe(ditele_df)

    st.subheader('(備考)予測データ')
    st.write('2022-12-31までを学習に使用しました')
    st.write('予測開始日時：'+ sim_start_date)
    plot_chart2(sim_data)
    st.dataframe(sim_data)