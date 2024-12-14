---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
    jupytext_version: '1.11.5'
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Prediksi Harga Saham


### Latar Belakang

<p style="text-indent: 50px; text-align: justify;">
Pasar saham merupakan salah satu indikator penting dalam mengukur kesehatan ekonomi suatu negara. Salah satu saham yang menarik perhatian investor di Indonesia adalah saham PT Telekomunikasi Indonesia Tbk (Telkom), yang merupakan perusahaan besar di sektor telekomunikasi. Fluktuasi harga saham Telkom dipengaruhi oleh berbagai faktor, termasuk kinerja perusahaan, kondisi ekonomi, serta faktor eksternal global dan lokal. Oleh karena itu, kemampuan untuk memprediksi harga saham menjadi penting bagi para investor dan analis pasar.
Pemanfaatan teknologi untuk memprediksi harga saham dapat memberikan keuntungan besar dalam pengambilan keputusan investasi. Dengan menggunakan data historis harga saham yang tersedia selama lima tahun, model prediksi berbasis pembelajaran mesin dapat memberikan perkiraan yang lebih akurat mengenai arah pergerakan harga saham Telkom di masa depan. Penelitian ini bertujuan untuk mengembangkan model prediksi yang dapat membantu para investor dalam merencanakan strategi investasi yang lebih efektif dan efisien.
</p>

### Rumusan Masalah

<p style="text-indent: 50px; text-align: justify;">
Bagaimana prediksi harga saham Telkom dapat meningkatkan keputusan investasi yang lebih menguntungkan bagi investor?
Apa metode yang paling efektif dalam memprediksi pergerakan harga saham Telkom untuk mendukung optimalisasi portofolio saham?
Bagaimana prediksi harga saham Telkom dapat membantu mengurangi risiko investasi di pasar saham?</p>

### Tujuan 

<p style="text-indent: 50px; text-align: justify;">
Meningkatkan Keputusan Investasi: Membantu investor membuat keputusan investasi yang lebih tepat terkait saham Telkom melalui prediksi harga yang akurat.
Optimalisasi Portofolio Saham: Memungkinkan manajer portofolio untuk mengalokasikan aset dengan lebih efisien, berdasarkan prediksi harga saham Telkom.
Meminimalkan Risiko: Menyediakan prediksi harga yang dapat mengurangi risiko kerugian bagi investor dalam perdagangan saham Telkom.
</p>


### Sumber Data 
<p style="text-indent: 50px; text-align: justify;">Dataset ini berasal dari Kaggle yang merupakan dataset saham telkom selama 5 tahun.  </p>

#### a. Data Preparation
```{code-cell} python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```
#### b. Data Wrangling
```{code-cell} python
data_df = pd.DataFrame(pd.read_csv("https://raw.githubusercontent.com/mellychandrawardani/mellychandrawardani/main/data_saham.csv"))
print(data_df.head())
```

<p style="text-indent: 50px; text-align: justify;"> 
Data mencakup informasi tanggal, harga pembukaan, tertinggi, terendah, penutupan, penutupan yang disesuaikan, dan volume perdagangan. Tanggal mencerminkan waktu transaksi, sementara harga pembukaan menunjukkan nilai saham saat pasar dibuka, sedangkan harga tertinggi dan terendah memberikan gambaran tentang fluktuasi harga dalam sehari. Harga penutupan mencerminkan nilai akhir pada hari itu, sedangkan penutupan yang disesuaikan memperhitungkan faktor-faktor seperti dividen dan pemecahan saham. Volume perdagangan menunjukkan jumlah saham yang diperdagangkan, yang dapat mengindikasikan minat investor. Dengan menganalisis data ini, investor dapat mengevaluasi tren, volatilitas, dan aktivitas pasar untuk membuat keputusan investasi yang lebih baik.</p>

#### c. Data Understanding
```{code-cell} python
# Memeriksa apakah kolom 'Date' ada
if 'Date' not in data_df.columns:
    raise KeyError("Kolom 'Date' tidak ditemukan. Periksa nama kolom dalam dataset.")

# Mengatur kolom 'Date' sebagai indeks dan mengubah formatnya
data_df['Date'] = pd.to_datetime(data_df['Date'])
data_df.set_index('Date', inplace=True)

# Memastikan kolom yang diperlukan ada
required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
data_df = data_df[required_columns]

# Menghapus missing values
data_df.dropna(inplace=True)
```
<p style="text-indent: 50px; text-align: justify;"> mempersiapkan dan membersihkan dataset data saham. Pertama, ia memeriksa keberadaan kolom 'Date' dan mengeluarkan kesalahan jika kolom tersebut tidak ditemukan. Selanjutnya, kolom 'Date' diubah menjadi format datetime dan diatur sebagai indeks DataFrame, memungkinkan operasi berbasis waktu yang lebih mudah. Kode ini kemudian memastikan bahwa hanya kolom yang diperlukan, yaitu 'Open', 'High', 'Low', 'Close', 'Adj Close', dan 'Volume', yang tersisa dalam DataFrame. Terakhir, baris dengan nilai hilang dihapus untuk memastikan data yang bersih dan lengkap, sehingga siap untuk analisis lebih lanjut.</p>

```{code-cell} python
data_df.plot()
```
```{code-cell} python
data_df.info()  # Menampilkan informasi umum mengenai DataFrame, termasuk jumlah entri, jumlah kolom, tipe data setiap kolom, dan apakah ada nilai kosong (non-null).
print('Ukuran data ', data_df.shape)  # Menampilkan ukuran data dalam format tuple (jumlah baris, jumlah kolom).
data_df.dtypes  # Menampilkan tipe data dari setiap kolom dalam DataFrame.
```

##### missing value
```{code-cell} python
# Mencari Missing Value
data_df.isnull().sum()
```
##### korelasi antar fitur
```{code-cell} python
import seaborn as sns
# Menghitung korelasi antar fitur untuk subset yang diinginkan
features = data_df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
correlation_matrix = features.corr()

# Menampilkan matriks korelasi
print("Matriks Korelasi:")
print(correlation_matrix)

# Menggambar heatmap untuk visualisasi korelasi
plt.figure(figsize=(10, 6))
plt.title("Heatmap Korelasi antar Fitur")
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()
```
<p style="text-indent: 50px; text-align: justify;">Hasil korelasi antar fitur menunjukkan hubungan linear antara pasangan fitur dalam data. Nilai korelasi berkisar dari -1 hingga 1, di mana nilai mendekati 1 menunjukkan korelasi positif yang kuat, mendekati -1 menunjukkan korelasi negatif yang kuat, dan mendekati 0 menunjukkan korelasi lemah atau tidak ada hubungan. Dari matriks ini, fitur "Open", "High", "Low", dan "Close" memiliki korelasi positif yang sangat kuat satu sama lain (di atas 0.99), menunjukkan bahwa pergerakan harga cenderung sejalan. Korelasi "Adj Close" juga tinggi terhadap fitur harga lainnya (sekitar 0.94–0.96), mengindikasikan hubungan yang signifikan dengan harga penutupan setelah penyesuaian. Sebaliknya, fitur "Volume" memiliki korelasi negatif lemah terhadap semua fitur harga (sekitar -0.29 hingga -0.34), yang menunjukkan bahwa volume perdagangan tidak secara langsung berkaitan dengan fluktuasi harga.</p>

#### d. Preprocessing Data
##### sliding windows
<p style="text-indent: 50px; text-align: justify;">Sliding window adalah teknik yang digunakan dalam pemrograman dan analisis data untuk mengatasi masalah yang melibatkan subarray atau subsequence dari data. Teknik ini memungkinkan kita untuk memproses elemen dalam array atau daftar secara efisien dengan menggunakan dua pointer atau indeks yang bergerak di sepanjang data.p>

```{code-cell} python
# Fungsi untuk membuat sliding windows
def create_sliding_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Menyiapkan data untuk sliding windows
window_size = 3  # Ukuran window
data_values = data_df[['Close', 'High']].values  # Menggunakan kolom 'Close' dan 'High' untuk prediksi
X, y = create_sliding_windows(data_values, window_size)

# Membuat DataFrame untuk hasil sliding windows
sliding_window_df = pd.DataFrame(X.reshape(X.shape[0], -1), columns=[f'Close_t-{window_size-i}' for i in range(window_size)] + [f'High_t-{window_size-i}' for i in range(window_size)])
sliding_window_df['Target_Close_t'] = y[:, 0]  # Target Close
sliding_window_df['Target_High_t'] = y[:, 1]  # Target High

# Menampilkan hasil sliding windows
print(sliding_window_df.head())
```

##### Normalisasi data
<p style="text-indent: 50px; text-align: justify;"> Normalisasi data adalah proses mengubah skala atau rentang nilai dari dataset sehingga semua fitur memiliki skala yang seragam, yang sangat penting dalam analisis data dan machine learning. Tujuannya adalah untuk menghindari dominasi fitur dengan rentang nilai yang lebih besar, meningkatkan konvergensi model dalam algoritma optimisasi, serta mempermudah visualisasi dan perbandingan antar fitur. Dengan normalisasi, model dapat beroperasi lebih efektif dan akurat, karena setiap fitur berkontribusi secara seimbang tanpa bias yang disebabkan oleh skala yang berbeda. Metode umum normalisasi termasuk Min-Max Scaling dan Z-Score Normalization, yang membantu dalam preprocessing data sebelum analisis lebih lanjut.</p>

```{code-cell} python
# Inisialisasi scaler untuk fitur (input) dan target (output)
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Normalisasi fitur (Close_t-3, Close_t-2, Close_t-1, High_t-3, High_t-2, High_t-1)
df_features_normalized = pd.DataFrame(
    scaler_features.fit_transform(sliding_window_df.iloc[:, :-2]),  # Ambil semua kolom kecuali target
    columns=sliding_window_df.columns[:-2],  # Nama kolom tanpa target
    index=sliding_window_df.index
)

# Normalisasi target (Target_Close_t dan Target_High_t)
df_target_normalized = pd.DataFrame(
    scaler_target.fit_transform(sliding_window_df[['Target_Close_t', 'Target_High_t']]),
    columns=['Target_Close_t', 'Target_High_t'],
    index=sliding_window_df.index
)

# Gabungkan kembali dataframe yang sudah dinormalisasi
df_normalized = pd.concat([df_features_normalized, df_target_normalized], axis=1)

# Menampilkan hasil normalisasi
print(df_normalized.head())

# Mengatur fitur (X) dan target (y)
X = df_normalized[['Close_t-3', 'Close_t-2', 'Close_t-1', 'High_t-3', 'High_t-2', 'High_t-1']]
y = df_normalized[['Target_Close_t', 'Target_High_t']]  # Target adalah harga yang dinormalisasi

# Membagi data menjadi training dan testing (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print('===== Data Train =====')
print(X_train)

print('===== Data Testing ====')
print(X_test)

# Mengambil nilai tanggal dari indeks X_train dan X_test
dates_train = X_train.index
dates_test = X_test.index

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Plot data Close dan High dengan format tanggal di sumbu x
plt.figure(figsize=(14, 7))

# Plot Close
plt.plot(data_df.index, data_df['Close'], label='Close', linestyle='-', color='blue')

# Plot High
plt.plot(data_df.index, data_df['High'], label='High', linestyle='--', color='orange')

# Format sumbu x agar menampilkan tanggal
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Menampilkan label tanggal per bulan

plt.title('Close and High Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.grid(True)
plt.xticks(rotation=45)  # Putar label tanggal agar tidak tumpang tindih
plt.tight_layout()
plt.show()
```
#### e. Modelling 

##### regresi linier 

```{code-cell} python
# Membuat model regresi linier
model = LinearRegression()
model.fit(X_train, y_train)

# Memprediksi harga menggunakan model
y_pred = model.predict(X_test)

# Menghitung Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Membuat plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Garis identitas
plt.xlabel('Nilai Aktual')
plt.ylabel('Nilai Prediksi')
plt.title(f'Plot Nilai Aktual vs Prediksi\nMSE: {mse:.2f}')
plt.grid()
plt.show()
```
##### random forest
```{code-cell} python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Misalkan dataset kamu punya fitur X dan target y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Memprediksi harga menggunakan Random Forest
rf_y_pred = rf_model.predict(X_test)

# Menghitung Mean Squared Error untuk Random Forest
rf_mse = mean_squared_error(y_test, rf_y_pred)
print(f'Mean Squared Error (Random Forest): {rf_mse}')

# Membuat plot untuk perbandingan hanya Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_y_pred, alpha=0.5, label='Random Forest', color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Identity Line')  # Garis identitas
plt.xlabel('Nilai Aktual')
plt.ylabel('Nilai Prediksi')
plt.title(f'Plot Nilai Aktual vs Prediksi (Random Forest)\nMSE: {rf_mse:.2f}')
plt.legend()
plt.grid()
plt.show()
```

#### f. Testing data baru
```{code-cell} python
# Pastikan dataset sudah dimuat sebelumnya dalam variabel data_df

# Buat fitur lag berdasarkan kolom 'Close' dan 'High'
data_df['Lag_Close_1'] = data_df['Close'].shift(1)
data_df['Lag_Close_2'] = data_df['Close'].shift(2)
data_df['Lag_Close_3'] = data_df['Close'].shift(3)

data_df['Lag_High_1'] = data_df['High'].shift(1)
data_df['Lag_High_2'] = data_df['High'].shift(2)
data_df['Lag_High_3'] = data_df['High'].shift(3)

# Hapus baris dengan nilai NaN akibat lagging
data_df = data_df.dropna()

# Pisahkan fitur (X) dan target (y)
X = data_df[['Lag_Close_1', 'Lag_Close_2', 'Lag_Close_3', 
             'Lag_High_1', 'Lag_High_2', 'Lag_High_3']].values
y_close = data_df['Close'].values.reshape(-1, 1)  # Target untuk 'Close'
y_high = data_df['High'].values.reshape(-1, 1)   # Target untuk 'High'

# Normalisasi data
from sklearn.preprocessing import StandardScaler
scaler_features = StandardScaler()
scaler_target_close = StandardScaler()
scaler_target_high = StandardScaler()

X_normalized = scaler_features.fit_transform(X)
y_close_normalized = scaler_target_close.fit_transform(y_close)
y_high_normalized = scaler_target_high.fit_transform(y_high)

# Bagi data menjadi data latih dan uji (80% latih, 20% uji)
train_size = int(0.8 * len(X_normalized))
X_train, X_test = X_normalized[:train_size], X_normalized[train_size:]
y_close_train, y_close_test = y_close_normalized[:train_size], y_close_normalized[train_size:]
y_high_train, y_high_test = y_high_normalized[:train_size], y_high_normalized[train_size:]

# Inisialisasi model Random Forest Regressor untuk masing-masing target
from sklearn.ensemble import RandomForestRegressor
rf_model_close = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_high = RandomForestRegressor(n_estimators=100, random_state=42)

# Melatih model
rf_model_close.fit(X_train, y_close_train.ravel())
rf_model_high.fit(X_train, y_high_train.ravel())

# Fungsi untuk memprediksi harga Close dan High
def predict_stock_prices(lag_close, lag_high):
    # Gabungkan input lag menjadi array
    lag_data = np.array(lag_close + lag_high).reshape(1, -1)

    # Normalisasi data input
    lag_data_normalized = scaler_features.transform(lag_data)

    # Prediksi harga
    predicted_close_normalized = rf_model_close.predict(lag_data_normalized)
    predicted_high_normalized = rf_model_high.predict(lag_data_normalized)

    # Kembalikan prediksi ke skala asli
    predicted_close_original = scaler_target_close.inverse_transform(predicted_close_normalized.reshape(-1, 1))
    predicted_high_original = scaler_target_high.inverse_transform(predicted_high_normalized.reshape(-1, 1))

    return predicted_close_original[0, 0], predicted_high_original[0, 0]

# Contoh input data
lag_close = [4000.0, 3970.0, 3930.0]  # Close 1, 2, dan 3 hari sebelumnya
lag_high = [4040.0, 4030.0, 3970.0]   # High 1, 2, dan 3 hari sebelumnya

# Prediksi harga
predicted_close, predicted_high = predict_stock_prices(lag_close, lag_high)

# Tampilkan hasil prediksi
print(f"Prediksi harga 'Close' untuk hari berikutnya adalah: {predicted_close:.2f}")
print(f"Prediksi harga 'High' untuk hari berikutnya adalah: {predicted_high:.2f}")

```

#### Kesimpulan
<p style="text-indent: 50px; text-align: justify;">Kesimpulannya dari hasil mse kedua model, random forest mendapatkan hasil yang lebih baik daripada regresi linier karena nilai mse nya lebih kecil.</p>

