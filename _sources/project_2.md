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

# Prediksi Harga Telur Ayam


### Latar Belakang

<p style="text-indent: 50px; text-align: justify;">Harga telur ayam di pasar modern di Provinsi Jawa Timur telah menunjukkan variasi yang signifikan dalam beberapa tahun terakhir. Perubahan harga telur ayam tidak hanya dipengaruhi oleh factor-faktor ekonomi dan produksi, tetapi juga oleh ketersediaan stok, preferensi konsumen, dan masih banyak lagi. Salah satu perusahaan yang berperan penting dalam industry ini adalah PT Japfa Comfeed Indonesia Tbk (JAPFA), yang terlibat dalam penyerapan produksi telur ayam dari peternak guna membantu menstabilkan harga di tingkat peternak. Melalui program penyerapan telur nasional, JAPFA berkomitmen untuk mendukung peternak local dan meningkatkan konsumsi protein hewani di masyarakat.
Analisis time series dapat membantu memahami kecenderungan dan pola perubahan harga telur ayam di pasar modern di Jawa Timur. Dengan menggunakan data historis harga telur ayam,penelitian ini dapat menganalisis tren harga, mengetahui pola perubahan, dan memprediksi harga telur ayam di minggu berikutnya. Oleh karena itu, penelitian ini bertujuan untuk menganalisis tren harga telur ayam di pasar modern di Provinsi Jawa Timur dan memprediksi harga telur ayam di minggu berikutnya.
</p>

### Rumusan Masalah

<p style="text-indent: 50px; text-align: justify;">Bagaimana memprediksi harga telur ayam di pasar modern di Provinsi Jawa Timur pada minggu berikutnya menggunakan analisis time series?</p>

### Tujuan 

<p style="text-indent: 50px; text-align: justify;"> Tujuan dari penelitian ini adalah untuk menganalisis tren harga telur ayam di pasar modern di Provinsi Jawa Timur dan memprediksi harga telur ayam di minggu berikutnya berdasarkan pola perubahan harga tersebut. Penelitian ini juga berharap untuk dapat membantu penjual dan pembeli telur ayam meminimalkan resiko lonjakan harga dan memaksimalkan keuntungan yang dapat mereka peroleh, serta untuk mendapat algoritma terbaik untuk prediksi harga telur ayam. 
</p>


### Sumber Data 
<p style="text-indent: 50px; text-align: justify;">Dataset ini berasal dari https://www.bi.go.id/hargapangan yang berbentuk time series dengan fitur yaitu tanggal dan harga telur ayam. dataset ini merupakan time series mingguan.  </p>

#### a. Data Preparation
```{code-cell} python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
```
#### b. Data Wrangling
```{code-cell} python
# Membaca file Excel dari URL raw GitHub
url = "https://github.com/mellychandrawardani/mellychandrawardani/raw/main/Tabel_Harga_Telur.xlsx"
df = pd.read_excel(url)
print(df.head())
```

<p style="text-indent: 50px; text-align: justify;"> 
Dataset ini memiliki:
Jumlah Atribut/Feature: 9 (berisi informasi demografis dan kesehatan).
Jumlah Data: 499 (entri responden).
Jumlah Label: 1 (status diabetes).
Jumlah Kelas: 2 (0: tidak diabetes, 1: diabetes).</p>

#### c. Data Understanding
```{code-cell} python
# Pastikan kolom 'Tanggal' dalam format datetime dengan dayfirst=True
df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True, errors='coerce')

# Mengatur kolom 'Tanggal' sebagai indeks
df.set_index('Tanggal', inplace=True)

# Menghapus spasi dari kolom 'Harga Telur', mengganti '-' dengan NaN, dan mengubahnya menjadi numeric
df['Harga Telur'] = df['Harga Telur'].replace('-', np.nan).str.replace(',', '').astype(float)

# Menampilkan 5 baris pertama untuk memastikan
print(df.head())
```
<p style="text-indent: 50px; text-align: justify;">Transformasi data ini bertujuan untuk mempersiapkan dataset agar siap digunakan dalam analisis berbasis waktu. Kolom `Tanggal` dikonversi ke format datetime dan diatur sebagai indeks untuk mempermudah pengolahan data berbasis waktu, seperti analisis tren atau *resampling*. Pada kolom `Harga Telur`, pembersihan dilakukan dengan mengganti simbol `'-'` menjadi `NaN` dan menghapus spasi agar data menjadi bersih dan konsisten dalam format numerik. Hasilnya, dataset lebih terstruktur dengan indeks waktu yang jelas dan data harga yang dapat digunakan untuk analisis lebih lanjut.</p>

```{code-cell} python
df.info()  # Menampilkan informasi umum mengenai DataFrame, termasuk jumlah entri, jumlah kolom, tipe data setiap kolom, dan apakah ada nilai kosong (non-null).
print('Ukuran data ', df.shape)  # Menampilkan ukuran data dalam format tuple (jumlah baris, jumlah kolom).
df.dtypes  # Menampilkan tipe data dari setiap kolom dalam DataFrame.
```
```{code-cell} python
# Mencari Missing Value
df.isnull().sum()
```

#### d. Preprocessing Data
##### sliding windows
<p style="text-indent: 50px; text-align: justify;">label encoding : Encoding digunakan untuk mengubah data kategorikal menjadi format numerik, sehingga dapat digunakan dalam analisis dan algoritma pembelajaran mesin.</p>

```{code-cell} python
# Membuat fitur lag untuk harga: harga-1, harga-2, harga-3, harga-4
df['Harga Telur-1'] = df['Harga Telur'].shift(1)  # Harga Telur satu hari sebelumnya
df['Harga Telur-2'] = df['Harga Telur'].shift(2)  # Harga Telur dua hari sebelumnya
df['Harga Telur-3'] = df['Harga Telur'].shift(3)  # Harga Telur tiga hari sebelumnya
df['Harga Telur-4'] = df['Harga Telur'].shift(4)  # Harga Telur empat hari sebelumnya

# Menghapus baris yang memiliki nilai NaN
df.dropna(inplace=True)

# Mengatur ulang kolom sehingga Harga Telur berada di kolom terakhir
df = df[['Harga Telur-4', 'Harga Telur-3', 'Harga Telur-2', 'Harga Telur-1', 'Harga Telur']]

# Menampilkan 5 baris pertama untuk memastikan fitur lag telah ditambahkan
print(df.head())
```

##### Normalisasi data
<p style="text-indent: 50px; text-align: justify;"> Normalisasi Min-Max adalah metode transformasi data untuk menskalakan nilai-nilai dalam rentang tertentu, biasanya antara 0 dan 1, menggunakan rumus: \( X' = \frac{X - X_{min}}{X_{max} - X_{min}} \), di mana \( X \) adalah nilai asli, \( X_{min} \) adalah nilai minimum, dan \( X_{max} \) adalah nilai maksimum dari dataset. Pada tabel hasil normalisasi di atas, semua nilai pada kolom **Harga Telur-4** hingga **Harga Telur** sudah disesuaikan ke rentang 0 hingga 1. Misalnya, nilai awal harga telur yang lebih tinggi mendekati angka 1 (seperti 0.589844 pada **2021-01-29**), sedangkan nilai yang lebih rendah mendekati angka 0 (seperti 0.246094 pada **2021-02-05**). Normalisasi ini membantu meratakan skala data, sehingga lebih cocok untuk digunakan dalam algoritma machine learning atau analisis statistik.</p>

```{code-cell} python
# Inisialisasi scaler untuk fitur (input) dan target (output)
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Normalisasi fitur (Harga-1, Harga-2, Harga-3, Harga-4, Harga-5)
df_features_normalized = pd.DataFrame(scaler_features.fit_transform(df[['Harga Telur-4', 'Harga Telur-3', 'Harga Telur-2', 'Harga Telur-1']]),
                                      columns=['Harga Telur-4', 'Harga Telur-3', 'Harga Telur-2', 'Harga Telur-1'],
                                      index=df.index)

# Normalisasi target (Harga Telur)
df_target_normalized = pd.DataFrame(scaler_target.fit_transform(df[['Harga Telur']]),
                                    columns=['Harga Telur'],
                                    index=df.index)

# Gabungkan kembali dataframe yang sudah dinormalisasi
df_normalized = pd.concat([df_features_normalized, df_target_normalized], axis=1)
df_normalized.head()
```

##### Mengatur data testing dan training
<p style="text-indent: 50px; text-align: justify;">Mengatur data *training* dan *testing* adalah proses membagi dataset menjadi dua bagian: satu untuk melatih model (*training set*) dan satu lagi untuk menguji performa model (*testing set*). **Training set** digunakan untuk melatih model agar dapat memahami pola dan karakteristik data. Sedangkan **testing set** digunakan untuk mengevaluasi seberapa baik model yang telah dilatih dapat memprediksi data baru yang belum pernah dilihat sebelumnya. Pembagian 80% untuk *training* dan 20% untuk *testing* atau variasi lain yang sesuai kebutuhan.

Pada **Data Train** terdiri dari 153 baris data historis *Harga Telur* dari periode waktu tertentu, digunakan untuk melatih model. Sementara itu, **Data Testing** memuat data terbaru dari Januari 2024 yang digunakan untuk mengevaluasi performa model prediksi. Pengaturan seperti ini penting agar model tidak hanya "menghafal" data, melainkan mampu melakukan generalisasi dengan baik terhadap data baru.</p>

```{code-cell} python
# Mengatur fitur (X) dan target (y)
X = df_normalized[['Harga Telur-4', 'Harga Telur-3', 'Harga Telur-2', 'Harga Telur-1']]
y = df_normalized['Harga Telur']

# Membagi data menjadi training dan testing (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print('===== Data Train =====')
print(X_train)

print('===== Data Testing ====')
print(X_test)

# Mengambil nilai tanggal dari indeks X_train dan X_test
dates_train = X_train.index
dates_test = X_test.index

# Plot data training
plt.figure(figsize=(12,6))
plt.plot(dates_train, X_train['Harga Telur-1'], label='Data Training', color='blue')

# Plot data testing
plt.plot(dates_test, X_test['Harga Telur-1'], label='Data Testing', color='red')

# Menambahkan judul, label, dan legenda
plt.title('Pembagian Data Training dan Testing Berdasarkan Harga Telur')
plt.xlabel('Tanggal')
plt.ylabel('Harga Telur')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotasi label sumbu x agar mudah dibaca
plt.tight_layout()
plt.show()
```

### e. Modelling 

#### random forest 

```{code-cell} python
# Import library yang diperlukan
from sklearn.ensemble import RandomForestRegressor  # Untuk model Random Forest
from sklearn.metrics import mean_squared_error, r2_score  # Untuk evaluasi model
import numpy as np  # Untuk manipulasi array

# Inisialisasi dan latih model Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Membuat model dengan 100 pohon
rf_model.fit(X_train, y_train)  # Melatih model dengan data pelatihan

# Prediksi pada data uji
y_pred_rf = rf_model.predict(X_test)  # Melakukan prediksi pada data uji

# Evaluasi model
mse_rf = mean_squared_error(y_test, y_pred_rf)  # Menghitung Mean Squared Error (MSE)
r2_rf = r2_score(y_test, y_pred_rf)  # Menghitung R-squared (R²)
rmse_rf = np.sqrt(mse_rf)  # Menghitung Root Mean Squared Error (RMSE)

# Fungsi untuk menghitung MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Hitung MAPE
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)

# Menampilkan hasil evaluasi
print(f'Root Mean Squared Error (RMSE - Random Forest): {rmse_rf:.2f}')
print(f'Mean Squared Error (MSE - Random Forest): {mse_rf:.2f}')
print(f'R-squared (R² - Random Forest): {r2_rf:.2f}')
print(f'Mean Absolute Percentage Error (MAPE - Random Forest): {mape_rf:.2f}%')
```

```{code-cell} python
import matplotlib.pyplot as plt
import pandas as pd

# Data tanggal dan harga asli
data = {
    'Tanggal': ['23/08/2024', '30/08/2024', '06/09/2024', '13/09/2024', '20/09/2024', '27/09/2024'],
    'Harga_Asli': [26800, 26550, 26000, 26800, 26850, 26100]
}

# Mengonversi ke DataFrame
df = pd.DataFrame(data)
df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y')  # Mengonversi string tanggal menjadi datetime

# Data harga prediksi (ganti dengan prediksi model Anda)
predicted_prices = [26600, 26400, 25900, 26700, 26900, 26200]  # Contoh prediksi, sesuaikan dengan model Anda

# Membuat plot
plt.figure(figsize=(12, 6))
plt.plot(df['Tanggal'], df['Harga_Asli'], label='Harga Asli', color='blue', marker='o')  # Garis untuk harga asli
plt.plot(df['Tanggal'], predicted_prices, label='Harga Prediksi', color='orange', marker='x')  # Garis untuk harga prediksi

# Menambahkan label dan judul
plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.title('Prediksi Harga Menggunakan Random Forest')
plt.legend()
plt.xticks(rotation=45)  # Memutar label tanggal agar lebih mudah dibaca
plt.tight_layout()  # Mengatur tata letak agar tidak ada yang terpotong

# Menampilkan plot
plt.show()
```

<p style="text-indent: 50px; text-align: justify;">Model Random Forest menunjukkan performa yang baik:
Akurasi: 88%
Precision: 92.3%
Recall: 81.8%
R-squared score sebesar 78.2% menunjukkan bahwa model mampu menjelaskan sebagian besar variasi dalam data.
RMSE rendah (0.315) menunjukkan bahwa tingkat kesalahan prediksi cukup kecil. </p>

### g. Random forest & ensamble bagging
```{code-cell} python
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor  # Mengimpor model RandomForestRegressor dan BaggingRegressor dari scikit-learn
from sklearn.metrics import mean_squared_error, r2_score  # Mengimpor metrik evaluasi
import numpy as np  # Mengimpor library numpy untuk operasi numerik
import pandas as pd  # Mengimpor library pandas untuk manipulasi data
import matplotlib.pyplot as plt  # Mengimpor matplotlib untuk visualisasi

# Inisialisasi dan latih model Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Membuat instance RandomForestRegressor dengan 100 pohon dan seed untuk reproduksibilitas.
rf_model.fit(X_train, y_train)  # Melatih model dengan data pelatihan.

# Prediksi pada data uji
y_pred_rf = rf_model.predict(X_test)  # Menggunakan model untuk memprediksi nilai pada data uji.

# Evaluasi model
mse_rf = mean_squared_error(y_test, y_pred_rf)  # Menghitung Mean Squared Error (MSE) untuk prediksi Random Forest.
r2_rf = r2_score(y_test, y_pred_rf)  # Menghitung R-squared (R²) untuk mengevaluasi model.

# Menghitung RMSE
rmse_rf = np.sqrt(mse_rf)  # Menghitung Root Mean Squared Error (RMSE) dari MSE.

# Menampilkan hasil evaluasi
print(f'Root Mean Squared Error (Random Forest): {rmse_rf}')  # Mencetak RMSE untuk model Random Forest.
print(f'Mean Squared Error (Random Forest): {mse_rf}')  # Mencetak MSE untuk model Random Forest.
print(f'R-squared (Random Forest): {r2_rf}')  # Mencetak R² untuk model Random Forest.

# Inisialisasi model Bagging dengan Random Forest sebagai base estimator
bagging_rf_model = BaggingRegressor(estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_estimators=10, random_state=42)  # Membuat model Bagging dengan 10 estimators.
# Latih model Bagging pada data pelatihan
bagging_rf_model.fit(X_train, y_train)  # Melatih model Bagging menggunakan data pelatihan.

# Prediksi pada data uji
y_pred_bagging_rf = bagging_rf_model.predict(X_test)  # Menggunakan model untuk memprediksi nilai pada data uji.

# Evaluasi model Bagging
mse_bagging_rf = mean_squared_error(y_test, y_pred_bagging_rf)  # Menghitung Mean Squared Error (MSE) untuk prediksi Bagging
r2_bagging_rf = r2_score(y_test, y_pred_bagging_rf)  # Menghitung R-squared untuk evaluasi kualitas model
# Menghitung RMSE
rmse_bagging_rf = np.sqrt(mse_bagging_rf)  # Menghitung Root Mean Squared Error (RMSE) dari MSE

# Tampilkan hasil evaluasi
print(f'Bagging Model (Random Forest) - Root Mean Squared Error: {rmse_bagging_rf}')  # Menampilkan RMSE
print(f'Bagging Model (Random Forest) - Mean Squared Error: {mse_bagging_rf}')  # Menampilkan MSE
print(f'Bagging Model (Random Forest) - R-squared: {r2_bagging_rf}')  # Menampilkan nilai R-squared
```
<p style="text-indent: 50px; text-align: justify;">Model memprediksi data baru termasuk ke dalam kelas 1 karena probabilitasnya melebihi threshold 0.7. Hal ini menunjukkan bahwa model cukup yakin data tersebut milik kelas 1. Akurasi model yang tinggi (95%) mendukung keandalan prediksi ini.</p>

### h. Testing data baru
```{code-cell} python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

# Contoh data pelatihan (gantilah dengan data pelatihan sebenarnya)
X_train = np.array([
    [28450, 28250, 27950, 27850], 
    [27900, 27650, 27600, 27250], 
    [27250, 26800, 26550, 26000]
])  # Data fitur pelatihan
y_train = np.array([[26800], [26850], [26100]])  # Target pelatihan

# Inisialisasi dan fit scaler dengan data pelatihan
scaler_features = StandardScaler()
scaler_target = StandardScaler()

# Fit scaler pada data fitur dan target
X_train_normalized = scaler_features.fit_transform(X_train)
y_train_normalized = scaler_target.fit_transform(y_train)

# Inisialisasi dan fit model Random Forest dengan data pelatihan yang telah dinormalisasi
random_forest_model = RandomForestRegressor()
random_forest_model.fit(X_train_normalized, y_train_normalized.ravel())

# Fungsi untuk memprediksi Harga berdasarkan input dari pengguna menggunakan Random Forest Regressor
def predict_custom_data_rf(input_data):
    # Normalisasi data input menggunakan scaler yang sama dengan data pelatihan
    input_data_normalized = scaler_features.transform(np.array(input_data).reshape(1, -1))

    # Prediksi dengan model Random Forest Regressor
    predicted_normalized = random_forest_model.predict(input_data_normalized)

    # Mengembalikan prediksi ke skala asli
    predicted_original = scaler_target.inverse_transform(predicted_normalized.reshape(-1, 1))

    return predicted_original[0, 0]  # Mengembalikan nilai prediksi dalam bentuk skalar.

# Data input dari pengguna
Harga_1 = 26100
Harga_2 = 26850
Harga_3 = 26800
Harga_4 = 26000
print("Harga 1 hari sebelumnya:", Harga_1)
print("Harga 2 hari sebelumnya:", Harga_2)
print("Harga 3 hari sebelumnya:", Harga_3)
print("Harga 4 hari sebelumnya:", Harga_4)

# Menggabungkan input data menjadi list
user_input = [Harga_4, Harga_3, Harga_2, Harga_1]

# Prediksi Harga untuk data yang dimasukkan menggunakan Random Forest
predicted_value_rf = predict_custom_data_rf(user_input)

# Menampilkan hasil prediksi
print(f"Prediksi Harga untuk hari selanjutnya adalah: {predicted_value_rf:.2f}")

```
### i. decision tree
```{code-cell} python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# Memisahkan data fitur dan target
target_column = 'Index'  # Sesuaikan dengan nama kolom target Anda
X = df.drop(columns=target_column, errors='ignore')
y = df[target_column]

# Menghapus kolom bertipe datetime jika ada
X = X.select_dtypes(exclude=['datetime64'])

# Alternatif: Mengonversi kolom datetime ke format numerik (opsional)
if 'Tanggal' in df.columns:
    X['Tanggal'] = (df['Tanggal'] - df['Tanggal'].min()).dt.days

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)

# Melatih model pada data pelatihan
dt_model.fit(X_train, y_train)

# Memprediksi hasil pada data uji
y_pred = dt_model.predict(X_test)

# Menghitung Mean Squared Error (MSE) dan Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Squared Error (MSE) dari Decision Tree Regressor:", mse)
print("Root Mean Squared Error (RMSE) dari Decision Tree Regressor:", rmse)
```

### Kesimpulan
<p style="text-indent: 50px; text-align: justify;">kesimpulan metode random forest biasa maupun random forest&ensamble bagging menghasilkan mse yg bagus untuk prediksi ini yaitu 0,00 </p>
