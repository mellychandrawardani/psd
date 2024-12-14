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

# Prediksi Penyakit Diabetes


## Pendahuluan 

### Latar Belakang

<p style="text-indent: 50px; text-align: justify;">Diabetes merupakan salah satu penyakit kronis yang terus berkembang dan dapat menyebabkan berbagai komplikasi serius jika tidak terdeteksi sejak dini. Dengan semakin meningkatnya jumlah penderita diabetes, prediksi risiko menjadi langkah krusial untuk mencegah perkembangan penyakit dan mengurangi beban biaya perawatan kesehatan jangka panjang. Pendekatan berbasis prediksi menggunakan faktor gaya hidup dan data klinis memungkinkan klinik kesehatan untuk memproyeksikan kemungkinan seorang pasien mengembangkan diabetes di masa depan. Dengan begitu, tindakan preventif dan intervensi medis yang lebih tepat dapat direncanakan. Sistem prediksi ini juga dapat meningkatkan efisiensi dalam pemanfaatan sumber daya klinik dan mempercepat pelayanan kesehatan preventif.</p>

### Rumusan Masalah

<p style="text-indent: 50px; text-align: justify;">1. Bagaimana cara membuat sistem yang dapat memprediksi risiko diabetes dengan akurat?  
2. Bagaimana hasil prediksi bisa membantu layanan kesehatan mencegah diabetes lebih efektif?  
3. Apa faktor yang paling mempengaruhi penyakit diabetes?</p>

### Tujuan 

<p style="text-indent: 50px; text-align: justify;"> 
1. Membuat sistem yang bisa memprediksi risiko diabetes dengan data kesehatan dan gaya hidup.  
2. Membantu layanan kesehatan dalam memberikan tindakan pencegahan lebih cepat dan tepat.  
3. Mengetahui faktor apa yang paling mempengaruhi diabetes.

</p>


### Sumber Data 
<p style="text-indent: 50px; text-align: justify;">Dataset ini berasal dari Kaggle dan berisi informasi mengenai data diabetes dengan berbagai fitur yang relevan untuk analisis kesehatan. Data ini akan digunakan untuk membangun model prediksi risiko diabetes berdasarkan kolom-kolom yang tersedia.Dataset yang digunakan ini berasal dari file Excel dengan informasi berikut:
gender: Tipe object (menunjukkan jenis kelamin responden: pria atau wanita).
age: Tipe int (usia responden dalam tahun).
hypertension: Tipe int (indikator apakah responden memiliki hipertensi: 0 untuk tidak, 1 untuk ya).
heart_disease: Tipe int (indikator apakah responden memiliki penyakit jantung: 0 untuk tidak, 1 untuk ya).
smoking_history: Tipe object (informasi mengenai riwayat merokok responden: 'never', 'current', 'former', atau 'not current').
bmi: Tipe float64 (Indeks Massa Tubuh dalam kg/m²).
HbA1c_level: Tipe float64 (level HbA1c dalam persen, indikator kontrol gula darah).
blood_glucose_level: Tipe float64 (level glukosa darah dalam mg/dL).
diabetes: Tipe int (indikator apakah responden menderita diabetes: 0 untuk tidak, 1 untuk ya).
Data ini memberikan gambaran komprehensif mengenai faktor-faktor yang dapat mempengaruhi risiko diabetes pada individu.</p>

#### a. Data Preparation
```{code-cell} python
# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.preprocessing import LabelEncoder

# Visualization
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
```
#### b. Data Wrangling
```{code-cell} python
data_df = pd.DataFrame(pd.read_excel("https://raw.githubusercontent.com/mellychandrawardani/mellychandrawardani/main/data_diabetes.xlsx"))
print(data_df.head())
```

<p style="text-indent: 50px; text-align: justify;"> 
Dataset ini memiliki:
Jumlah Atribut/Feature: 9 (berisi informasi demografis dan kesehatan).
Jumlah Data: 499 (entri responden).
Jumlah Label: 1 (status diabetes).
Jumlah Kelas: 2 (0: tidak diabetes, 1: diabetes).</p>

```{code-cell} python
kelas = "diabetes"

print(f"Jumlah Atribut/Feature : {len(data_df.axes[1])}")
print(f"Jumlah Data : {len(data_df.axes[0])}")
print(f"Jumlah Label : 1 [{kelas}]")
print(f"Jumlah Kelas : {len(data_df[kelas].unique())} {data_df[kelas].unique()}")
df.head()
```

#### c. Exploratory Data Anaysis
cek tipe dataset, cek missing value, cek duplikat data
```{code-cell} python
# Mengecek tipe dataset
data_df.info()
# Mengecek jumlah missing value di setiap kolom
missing_values = data_df.isnull().sum()
# Menampilkan hasil
print("Jumlah missing value per kolom:")
print(missing_values)
# Menampilkan persentase missing value
total_rows = len(data_df)
missing_percentage = (missing_values / total_rows) * 100

print("\nPersentase missing value per kolom:")
print(missing_percentage)
#duplikat data
data_df.duplicated().sum()
```
<p style="text-indent: 50px; text-align: justify;">Hasil analisis missing value dalam dataset menunjukkan bahwa:
Jumlah Missing Value per Kolom: Semua kolom memiliki 0 missing value, artinya tidak ada data yang hilang dalam setiap atribut.
Persentase Missing Value per Kolom: Semua kolom juga menunjukkan persentase 0.0%, yang berarti tidak ada nilai yang hilang dari keseluruhan dataset. Dengan tidak adanya missing value, analisis dan pemodelan dapat dilakukan tanpa harus menangani data yang hilang, sehingga meningkatkan kualitas dan akurasi hasil analisis.</p>

#### d. Preprocessing Data

<p style="text-indent: 50px; text-align: justify;">label encoding : Encoding digunakan untuk mengubah data kategorikal menjadi format numerik, sehingga dapat digunakan dalam analisis dan algoritma pembelajaran mesin.</p>

```{code-cell} python
# label encoder
le = LabelEncoder()
le_copy = data_df.apply(lambda x: x.unique())
le_copy_encode = le_copy.apply(lambda x: le.fit_transform(x))

print(f"Atribut diabetes \n{le_copy}\n")
print(f"Atribut diabetes Encode \n{le_copy_encode}\n")
```
```{code-cell} python
data_df = data_df.apply(lambda x: le.fit_transform(x))
data_df
```

##### Korelasi Antar Fitur
<p style="text-indent: 50px; text-align: justify;"> Korelasi antar fitur digunakan untuk memahami hubungan antara variabel dalam dataset. Hasilnya: "HbA1c_level" dan "blood_glucose_level" menunjukkan korelasi positif yang signifikan. "blood_glucose_level" berkorelasi dengan target "diabetes", mendukung fakta bahwa peningkatan kadar gula darah berkaitan erat dengan risiko diabetes. Fitur-fitur lain seperti "gender" atau "smoking_history" tampaknya tidak berpengaruh besar terhadap fitur lain dalam dataset ini.</p>

```{code-cell} python
corr_data_df = data_df.copy()
corr = corr_data_df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=0, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(corr_data_df.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(corr_data_df.columns)
ax.set_yticklabels(corr_data_df.columns)
plt.show()
```

##### Seleksi Fitur
<p style="text-indent: 50px; text-align: justify;">Seleksi fitur adalah proses memilih subset fitur yang paling relevan dari dataset untuk digunakan dalam model pembelajaran mesin. Hasil seleksi ini menunjukkan bahwa fitur-fitur yang dipilih memiliki keterkaitan yang lebih signifikan terhadap prediksi diabetes, yaitu: Kadar glukosa darah (blood_glucose_level) dan HbA1c_level sebagai indikator langsung. Faktor risiko klinis, seperti BMI dan heart_disease. Faktor demografis, seperti age dan gender.
Gaya hidup, seperti smoking_history.</p>

```{code-cell} python
main_df = pd.concat([data_df[data_df.columns[0:2]], data_df[data_df.columns[3:]]], axis=1)
features_df = main_df[main_df.columns[0:-1]]
labels_df = main_df["diabetes"]
main_df
```

### e. Modelling 

#### random forest 

```{code-cell} python
train_features, test_features, train_labels, test_labels = train_test_split(features_df, labels_df, test_size = 0.25, random_state=1)
```
```{code-cell} python
rf = RandomForestClassifier()
# Train the model on training data
rf.fit(train_features, train_labels);
```
```{code-cell} python
# Create predict for model
predictions = rf.predict(test_features)

# Calculate performance metrics for regression
mse = mean_squared_error(test_labels, predictions)
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(test_labels, predictions)

print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
```
```{code-cell} python
# Calculate accuracy precision and recall
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
```
```{code-cell} python
# Create the confusion matrix
cm = confusion_matrix(test_labels, predictions)

ConfusionMatrixDisplay(confusion_matrix=cm).plot();
```
```{code-cell} python
# rf = RandomForestClassifier()
# rf.fit(train_features, train_labels)  # Fit the model

for i in range(3):
  # Pick one tree from the forest, e.g., the first tree (index 0)
  tree_to_plot = rf.estimators_[i]

  name_class = [str(c) for c in tree_to_plot.classes_]

  # Plot the decision tree
  plt.figure(figsize=(30, 20))
  plot_tree(tree_to_plot, feature_names=features_df.columns, class_names=name_class, filled=True, rounded=True, fontsize=10)
  plt.title("Decision Tree from Random Forest")
  plt.show()
  ```

<p style="text-indent: 50px; text-align: justify;">Model Random Forest menunjukkan performa yang baik:
Akurasi: 88%
Precision: 92.3%
Recall: 81.8%
R-squared score sebesar 78.2% menunjukkan bahwa model mampu menjelaskan sebagian besar variasi dalam data.
RMSE rendah (0.315) menunjukkan bahwa tingkat kesalahan prediksi cukup kecil. </p>

### f. Feature Important
```{code-cell} python
# Membagi data menjadi fitur dan label
X = data_df.drop('diabetes', axis=1)
y = data_df['diabetes']

# Mengkodekan variabel kategorikal jika diperlukan
X = pd.get_dummies(X, drop_first=True)

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat dan melatih model Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Menghitung feature importance
importances = model.feature_importances_

# Membuat DataFrame untuk feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Menampilkan DataFrame feature importance
print(feature_importance_df)

# Menampilkan feature importance dalam bentuk grafik
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
```

<p style="text-indent: 50px; text-align: justify;">Dari hasil fitur penting, fitur yang paling berpengaruh terhadap diabetes adalah HbA1c_level dengan nilai 0.400882, menunjukkan kontribusi signifikan dalam memprediksi risiko diabetes. Diikuti oleh blood_glucose_level yang memiliki nilai 0.273806, menandakan bahwa kadar glukosa darah juga berperan besar. Selanjutnya, age dengan nilai 0.131057 menunjukkan bahwa usia merupakan faktor penting, sementara fitur lainnya seperti BMI dan riwayat merokok memiliki pengaruh yang lebih kecil. Ini menunjukkan bahwa kontrol gula darah dan kesehatan metabolik adalah indikator utama risiko diabetes.</p>

### g. Testing Data Baru
```{code-cell} python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Dataframe yang di-load
# data_df adalah dataset yang kamu miliki
# Pastikan sudah ada kolom `diabetes` sebagai target
# Misalnya: data_df = pd.read_csv('path_to_file.csv')

# Membagi data menjadi fitur dan label
X = data_df.drop('diabetes', axis=1)
y = data_df['diabetes']

# Mengkodekan variabel kategorikal
X = pd.get_dummies(X, drop_first=True)

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat dan melatih model Random Forest
model = RandomForestClassifier(
    random_state=42,
    min_samples_split=5,    # Mencegah overfit
    min_samples_leaf=3      # Meningkatkan generalisasi
)
model.fit(X_train, y_train)

# Menambahkan satu data baru untuk pengujian
data_baru = pd.DataFrame({
    'gender': ['Female'],
    'age': [50],
    'hypertension': [0],
    'heart_disease': [0],
    'smoking_history': ['never'],
    'bmi': [26.5],
    'HbA1c_level': [10],
    'blood_glucose_level': [200]
})

# Mengkodekan variabel kategorikal pada data baru
data_baru_encoded = pd.get_dummies(data_baru, drop_first=True)

# Memastikan data baru memiliki kolom yang sama dengan X_train
for column in X.columns:
    if column not in data_baru_encoded.columns:
        data_baru_encoded[column] = 0

data_baru_encoded = data_baru_encoded[X.columns]  # Menyusun ulang kolom sesuai dengan X

# Melakukan prediksi pada data pengujian
y_pred = model.predict(X_test)

# Melakukan prediksi probabilitas pada data baru
y_pred_proba = model.predict_proba(data_baru_encoded)

# Menggunakan ambang batas probabilitas 0.7 untuk memutuskan prediksi
threshold = 0.7
y_pred_baru = (y_pred_proba[:, 1] >= threshold).astype(int)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy:.2f}')

# Menampilkan confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Menampilkan hasil prediksi untuk data baru
print(f'Prediksi untuk data baru: {y_pred_baru[0]} (Probabilitas: {y_pred_proba[0]})')
print(f'Dengan threshold {threshold}, hasil prediksi adalah: {y_pred_baru[0]}')

# Menampilkan confusion matrix dalam bentuk grafik
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
```
<p style="text-indent: 50px; text-align: justify;">Model memprediksi data baru termasuk ke dalam kelas 1 karena probabilitasnya melebihi threshold 0.7. Hal ini menunjukkan bahwa model cukup yakin data tersebut milik kelas 1. Akurasi model yang tinggi (95%) mendukung keandalan prediksi ini.</p>

### Kesimpulan
<p style="text-indent: 50px; text-align: justify;">kesimpulan metode random forest tersebut sudah baik karena menghasilkan mse yang kecil yaitu 0.048. fitur yg paling berpengaruh yaitu hba1c_level serta blood_glucose_level.</p>
