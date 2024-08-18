


import numpy as np
import pandas as pd
import talib as ta
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import requests

# CoinGecko API'den veri çekme
def fetch_data_from_coingecko(coin_id='ethereum', vs_currency='usd', days=90):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {'vs_currency': vs_currency, 'days': days}
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# CoinGecko'dan Ethereum fiyat verilerini çekme
eth_data = fetch_data_from_coingecko('ethereum', 'usd', days=365)

# Veri seti yükleme ve CoinGecko verileriyle birleştirme
local_data = pd.read_csv('data.csv')  # Örnek yerel veri dosyası
data = pd.merge(local_data, eth_data, left_index=True, right_index=True, how='outer')
data.fillna(method='ffill', inplace=True)  # Eksik verileri doldurma

# Kapanış fiyatlarını kullanma
close_prices = data['price'].values
close_prices = close_prices.reshape(-1, 1)

# MinMaxScaler kullanarak veriyi normalize etme
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# RSI hesaplama (14 periyot)
data['RSI'] = ta.RSI(data['price'], timeperiod=14)

# MACD hesaplama
data['MACD'], data['MACD_signal'], data['MACD_hist'] = ta.MACD(data['price'], fastperiod=12, slowperiod=26, signalperiod=9)

# Özellikler ve hedefler oluşturma
X = []
y = []
window_size = 60

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# LSTM modeli oluşturma
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Öğrenme oranını artırarak modeli derleme
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Modeli eğitme
model.fit(X, y, epochs=25, batch_size=32)

# Tahmin yapma
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Sonuçları yazdırma
print(predicted_prices)

