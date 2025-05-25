# Laporan Proyek Machine Learning – Khansa Maritza A

## Domain Proyek

Penyakit jantung merupakan penyebab kematian tertinggi di dunia, dengan lebih dari 17,9 juta kasus setiap tahun (WHO, 2023). Gaya hidup modern, stres, dan dampak COVID-19 turut memperburuk kondisi ini, bahkan di usia produktif (Sahoo & Jeripothula, 2020). Deteksi dini menjadi penting untuk mencegah komplikasi serius, namun prosesnya masih kompleks dan membutuhkan sistem prediksi yang akurat.

Dengan dukungan data kesehatan dan perkembangan machine learning (ML), prediksi risiko penyakit jantung kini dapat dilakukan lebih efektif. Penelitian oleh Gandla et al. (2023) dan Sahoo & Jeripothula (2020) menunjukkan bahwa algoritma seperti SVM, Random Forest, dan Logistic Regression mampu menghasilkan prediksi yang baik pada dataset UCI.

Proyek ini bertujuan membangun model ML untuk memprediksi risiko penyakit jantung berdasarkan data klinis pasien sebagai upaya deteksi dini dan dukungan pengambilan keputusan medis.

Referensi:
- World Health Organization. (2023). Cardiovascular diseases (CVDs). Online at https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds), accessed 24 May 2025
- Gandla, V. R., Mallela, D. V., & Chaurasiya, R. (2023, June). Heart failure prediction using machine learning. In AIP Conference Proceedings (Vol. 2705, No. 1). AIP Publishing.
- Sahoo, P. K., & Jeripothula, P. (2020). Heart failure prediction using machine learning techniques. Available at SSRN 3759562.

## Business Understanding

### Problem Statements
- Bagaimana cara memprediksi potensi seorang pasien mengalami penyakit jantung berdasarkan data yang tersedia?
- Algoritma machine learning manakah yang paling optimal untuk digunakan dalam kasus prediksi penyakit jantung ini?

### Goals
- Menghasilkan model machine learning yang mampu memprediksi risiko penyakit jantung dengan tingkat akurasi yang tinggi.
- Mengidentifikasi model terbaik dari beberapa algoritma yang digunakan melalui evaluasi metrik performa.

### Solution statements
- Menerapkan beberapa algoritma klasifikasi, antara lain Logistic Regression, Random Forest, dan Support Vector Machine.
- Melakukan hyperparameter tuning guna meningkatkan performa masing-masing model.
- Menggunakan metrik evaluasi seperti F1-score, precision, recall, dan akurasi untuk memilih model terbaik.

## Data Understanding
**Sumber Dataset**: [Heart Failure Prediction Dataset - fedesoriano]([https://finance.yahoo.com/quote/TLKM.JK](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction))  
- Jumlah Data: 918 observasi
- Tidak ada data duplikat dan missing values
  
### Variabel-variabel pada Heart Failure Prediction Dataset
| **Kolom**      | **Tipe Data** | **Deskripsi**                                                        |
| -------------- | ------------- | -------------------------------------------------------------------- |
| Age            | int64         | Usia pasien                                                          |
| Sex            | object        | Jenis kelamin pasien (0 = perempuan, 1 = laki-laki)                  |
| ChestPainType  | object        | Tipe nyeri dada yang dialami pasien                                  |
| RestingBP      | int64         | Tekanan darah saat istirahat (mmHg)                                  |
| Cholesterol    | int64         | Kadar kolesterol dalam darah                                         |
| FastingBS      | int64         | Status gula darah puasa (1 jika ≥ 120 mg/dl, 0 jika < 120 mg/dl)     |
| RestingECG     | object        | Hasil elektrokardiogram saat istirahat                               |
| MaxHR          | int64         | Detak jantung maksimum saat berolahraga                              |
| ExerciseAngina | object        | Apakah pasien mengalami angina saat berolahraga (Y/N)                |
| Oldpeak        | float64       | Depresi segmen ST akibat olahraga dibandingkan saat istirahat        |
| ST_Slope      | object        | Kemiringan segmen ST saat puncak olahraga (Up/Flat/Down)             |
| HeartDisease   | int64         | Label target: 1 jika pasien menderita penyakit jantung, 0 jika tidak |


### Data Visualization
![image](https://github.com/user-attachments/assets/829b6dbd-c73f-4a85-a506-ea27393d930b)
Visualisasi data menggunakan boxplot dilakukan pada fitur numerik seperti Age, RestingBP, Cholesterol, MaxHR, dan Oldpeak. Tujuan dari visualisasi ini adalah untuk memahami distribusi data serta mengidentifikasi potensi outlier pada masing-masing fitur. Dengan memanfaatkan library Seaborn dan Matplotlib, dilakukan plotting secara bersamaan agar mempermudah perbandingan antar fitur secara visual.

## Data Preparation
### Menangani outliers
Kode clipping IQR dipakai untuk membatasi nilai ekstrem pada fitur numerik tanpa menghapus data.
```
for column in num_cols:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr
    df[column] = df[column].clip(lb, ub)
```
Alasan:
Metode ini menjaga distribusi data tetap stabil dan mencegah outlier mengganggu performa model.

### Encoding Fitur Kategori
Fitur kategorikal pada dataset dikonversi menggunakan metode One-Hot Encoding. Teknik ini mengubah setiap kategori pada fitur menjadi representasi fitur biner terpisah.
```
# List fitur kategorikal lo
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Lakukan One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features, drop_first=True, dtype=int)

df_encoded.head()
```
Alasan:
Agar model tidak menganggap fitur kategori sebagai data numerik yang berurutan, sehingga prediksi jadi lebih akurat.
### Train-Test-Split
Setelah data di-encode, fitur (X) dan target (y) dipisahkan. Dataset kemudian dibagi menjadi data latih (train) dan data uji (test) dengan proporsi 80:20.
```
# Pakai data yang udah di-encode
X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

print(f'Total jumlah sampel di dataset: {len(X)}')
print(f'Jumlah sampel di data latih (train): {len(X_train)}')
print(f'Jumlah sampel di data uji (test): {len(X_test)}')
```
Proporsi Data:
- Data latih (train): 734 sampel
- Data uji (test): 184 sampel

Alasan: Pembagian data ini penting untuk menghindari overfitting, di mana model hanya menghafal data latih tanpa mampu melakukan generalisasi pada data baru.

### Standarisasi 
Fitur numerik dinormalisasi menggunakan StandardScaler untuk menyamakan skala antar fitur, sehingga model dapat belajar lebih efektif tanpa bias terhadap nilai dengan skala yang lebih besar. 
```
numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']  # tanpa 'HeartDisease'

scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
```
Alasan: Mempercepat proses pelatihan dan menghindari bias model terhadap fitur dengan nilai besar.


## Modeling
Pada tahap ini, lima algoritma machine learning digunakan untuk membangun model klasifikasi risiko penyakit jantung. Proses pemodelan dilakukan dengan pendekatan hyperparameter tuning menggunakan RandomizedSearchCV, yang memungkinkan pencarian kombinasi parameter terbaik secara efisien.
Model dan Parameter
Berikut algoritma yang digunakan beserta parameter yang dituning:

**1.	Logistic Regression**
- Model: LogisticRegression(max_iter=10000, random_state=42)
- Parameter:
    - C: tingkat regularisasi, dicari dari distribusi uniform (0.01 hingga 10)
    - solver: metode optimisasi ('lbfgs' dan 'saga')

**2.	Random Forest**
- Model: RandomForestClassifier(random_state=42)
- Parameter:
    - n_estimators: jumlah pohon, antara 50 hingga 300
    - max_depth: kedalaman maksimum pohon (3-20)
    - min_samples_split: jumlah minimum sampel untuk split (2-10)
    - min_samples_leaf: jumlah minimum sampel pada daun (1-10)
    - max_features: metode pemilihan fitur ('sqrt', 'log2', None)
      
**3.	Gradient Boosting**
- Model: GradientBoostingClassifier(random_state=42)
- Parameter:
    - n_estimators: jumlah estimator (50-300)
    - learning_rate: kecepatan belajar dari distribusi uniform (0.01-0.3)
    - max_depth: kedalaman pohon (3-10)
      
**4.	AdaBoost**
- Model: AdaBoostClassifier(random_state=42)
- Parameter:
    - n_estimators: jumlah estimator (50-300)
    - learning_rate: tingkat pembelajaran (0.01-1.0)
      
**5.	Support Vector Machine (SVM)**
- Model: SVC(probability=True, random_state=42)
- Parameter:
    - C: parameter regularisasi dari distribusi uniform (0.1-10)
    - kernel: menggunakan kernel 'rbf'
    - gamma: 'scale' atau 'auto'

Keterangan: Semua model dilatih menggunakan teknik RandomizedSearchCV untuk menemukan kombinasi parameter terbaik.

### Kelebihan dan Kekurangan Setiap Algoritma
**1.	Random Forest**
- Kelebihan: Akurasi tinggi dan tahan terhadap overfitting. Mampu menangani fitur yang kompleks.
- Kekurangan: Proses pelatihan memerlukan sumber daya besar dan sulit diinterpretasikan.
  
**2.	AdaBoost**
- Kelebihan: Performa tinggi untuk metrik recall, cocok untuk mendeteksi kasus positif sebanyak mungkin.
- Kekurangan: Sensitif terhadap outlier dan noise. Bisa overfitting jika tidak dituning.
  
**3.	Logistic Regression**
- Kelebihan: Sederhana, cepat, dan interpretatif.
- Kekurangan: Kurang cocok untuk data dengan pola non-linear.
  
**4.	Gradient Boosting**
- Kelebihan: Presisi dan F1-score tinggi. Mampu mempelajari pola kompleks.
- Kekurangan: Latihannya lebih lambat dan rentan overfitting.
  
**5.	Support Vector Machine (SVM)**
- Kelebihan: Cocok untuk data berdimensi tinggi dan kelas yang terpisah dengan jelas.
- Kekurangan: Kurang efisien untuk dataset besar.
### Pemilihan Model Terbaik
Berdasarkan hasil evaluasi, AdaBoost dipilih sebagai model terbaik. Meskipun akurasinya sedikit di bawah Random Forest, AdaBoost memiliki recall tertinggi.

Alasan:
Dalam konteks prediksi penyakit jantung, mendeteksi pasien yang benar-benar berisiko lebih penting daripada sekadar meningkatkan akurasi keseluruhan. Oleh karena itu, performa recall yang tinggi menjadikan AdaBoost lebih tepat digunakan.

## Evaluation
Untuk mengukur performa model dalam klasifikasi risiko penyakit jantung, digunakan beberapa metrik evaluasi berikut:

![image](https://github.com/user-attachments/assets/16221fdd-158b-4119-bfa1-cd3a1f445657)

- **Accuracy**: Persentase prediksi yang benar dibandingkan dengan seluruh data yang diuji.
- **Precision**: Mengukur ketepatan model dalam memprediksi kelas positif (risiko penyakit). Precision penting ketika perlu menghindari terlalu banyak false positive.
- **Recall**: Mengukur seberapa banyak kasus positif yang berhasil dikenali oleh model. Metrik ini sangat penting dalam konteks kesehatan, karena bertujuan meminimalkan kasus yang tidak terdeteksi (false negative).
- **F1-Score**: Rata-rata gabungan dari precision dan recall yang mempertimbangkan keseimbangan antara keduanya.
- **Confusion Matrix**: Matriks 2x2 yang menggambarkan jumlah prediksi benar dan salah pada masing-masing kelas. Matriks ini membantu memahami secara detail jenis kesalahan yang dilakukan model.
  
### Hasil Evaluasi
Berdasarkan hasil training dan pengujian, performa masing-masing model adalah sebagai berikut:
| Model               | Accuracy   | Precision | Recall     | F1-score   |
| ------------------- | ---------- | --------- | ---------- | ---------- |
| Random Forest       | 0.8641     | 0.8545    | 0.9126     | 0.8826     |
| AdaBoost            | 0.8587     | 0.8407    | **0.9223** | 0.8796     |
| Logistic Regression | 0.8478     | 0.8378    | 0.9029     | 0.8692     |
| Gradient Boosting   | 0.8478     | 0.8440    | 0.8932     | 0.8679     |
| SVM                 | 0.8478     | 0.8378    | 0.9029     | 0.8692     |

Model Random Forest dan AdaBoost menunjukkan performa terbaik, khususnya pada recall dan F1-score, yang menjadi metrik utama dalam konteks klasifikasi penyakit. Model dengan recall tinggi lebih diutamakan karena lebih sedikit melewatkan kasus positif (penderita penyakit jantung).

### Analisis Confusion Matrix
![Untitled design](https://github.com/user-attachments/assets/b21bf3b2-8c50-4cfe-a8a4-de927ffbb9ca)
1. Logistic Regression menunjukkan kinerja yang cukup baik, dengan jumlah FN yang relatif rendah, meskipun jumlah FP masih cukup tinggi.

2. Random Forest memiliki jumlah False Negative terendah kedua setelah AdaBoost, serta True Positive tertinggi, menunjukkan bahwa model ini cukup andal dalam mengidentifikasi pasien berisiko. Hal ini terlihat dari nilai recall dan F1-score yang tinggi.

3. Gradient Boosting memiliki performa yang cukup seimbang, meskipun jumlah FN dan FP sedikit lebih tinggi dibanding Random Forest dan AdaBoost. Hal ini mengindikasikan model ini sedikit lebih longgar dalam klasifikasinya.

4. AdaBoost memiliki jumlah False Negative paling rendah (8 kasus), yang berarti model ini paling sedikit melewatkan pasien yang benar-benar berisiko. Ini menjadikannya sangat relevan untuk digunakan dalam konteks prediksi kesehatan, di mana kesalahan dalam mendeteksi kasus positif bisa berdampak serius.

5. Support Vector Machine (SVM) menunjukkan pola hasil yang identik dengan Logistic Regression, dengan jumlah FN dan FP yang sama. Artinya, dari sisi kesalahan klasifikasi, kinerjanya cukup mirip.

### Kesimpulan
Dilihat dari keseluruhan analisis dan mempertimbangkan pentingnya mengurangi False Negative dalam konteks kesehatan, **AdaBoost** menjadi kandidat terbaik untuk digunakan, diikuti oleh Random Forest. Kedua model ini menunjukkan kemampuan tinggi dalam mendeteksi pasien berisiko tanpa terlalu banyak menghasilkan prediksi false positif.
Meskipun akurasi Random Forest lebih tinggi, AdaBoost dipilih karena memiliki recall tertinggi dan false negative paling rendah. Hal ini penting dalam prediksi penyakit jantung karena lebih baik mendeteksi semua pasien berisiko daripada melewatkan yang positif.

### Evaluasi terhadap Goals Proyek
- Goals 1: Menghasilkan model machine learning yang mampu memprediksi risiko penyakit jantung dengan tingkat akurasi yang tinggi.
<br> ✅ Tercapai. Beberapa model berhasil mencapai akurasi di atas 84%, dengan performa terbaik oleh Random Forest (86.41%) dan AdaBoost (85.87%).

- Goals 2: Mengidentifikasi model terbaik dari beberapa algoritma yang digunakan melalui evaluasi metrik performa.
<br> ✅ Tercapai. Model AdaBoost diidentifikasi sebagai model terbaik berdasarkan kombinasi recall tertinggi dan FN terendah, menjadikannya solusi paling ideal untuk kasus prediksi risiko penyakit jantung.
