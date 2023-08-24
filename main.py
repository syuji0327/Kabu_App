import math
import numpy as np
import tqdm
import torch
import torch.nn as nn
import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
# import optuna.integration.lightgbm as lgb_optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
from sklearn import metrics
import streamlit as st

ACCESS_KEY_JSON = "kabuapp-d47e1f69fa4b.json"
SPREAD_SHEET_KEY = "14AcFe46sjuzkTuxk3T28hq4bhgEWRDJiLHL5ifHnctM"
SHEET_NAME = 'sony_2000_2023'

scopes = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive'
]

# Credentials 情報を取得
credentials = ServiceAccountCredentials.from_json_keyfile_name(ACCESS_KEY_JSON, scopes)

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

train_data.shape, test_data.shape

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

print(sim_data)


# Streamlit App
def main():
    # stleamlit タイトルとテキストを記入
    st.title(SHEET_NAME +'の株価')

    # データフレームの表示
    st.title('株価の推移グラフ')
    st.write(df)

    # グラフの表示
    st.write('終値と日時を表示')
    plot_chart(df)

    st.write('シミュレーションについて')
    sim_start_date = st.date_input('開始日時')
    sim_end_date = st.date_input('終了日時')

    st.write('開始日時：'+ sim_start_date)

    plot_chart2(sim_data)

    st.sidebar.title('シミュレーション設定')
    xmin=st.sidebar.number_input('初期所持金（円）：',0,100000000,0)
    ymax=st.sidebar.number_input('手数料（円）：',0,2000,0)

if __name__ == '__main__':
    main()