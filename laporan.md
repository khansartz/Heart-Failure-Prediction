# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek

Pada bagian ini, kamu perlu menuliskan latar belakang yang relevan dengan proyek yang diangkat.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding

### Problem Statements
- Bagaimana cara memprediksi apakah seorang pasien berpotensi mengalami penyakit jantung berdasarkan data klinis?
- Algoritma machine learning mana yang paling optimal untuk kasus prediksi penyakit jantung?

### Goals
- Menghasilkan model machine learning yang dapat memprediksi risiko penyakit jantung secara akurat.
- Mengidentifikasi model terbaik dari beberapa algoritma melalui evaluasi metrik performa.

### Solution statements
- Menerapkan beberapa algoritma klasifikasi seperti Logistic Regression, Random Forest, dan Support Vector Machine.
- Melakukan hyperparameter tuning untuk meningkatkan performa model.
- Menggunakan metrik evaluasi seperti F1-score, precision, recall, dan akurasi untuk memilih model terbaik.

## Data Understanding
**Sumber Dataset**: [Heart Failure Prediction Dataset - fedesoriano]([https://finance.yahoo.com/quote/TLKM.JK](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction))  

Jumlah Data: 918 observasi
Jumlah 

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- Age: Usia pasien
- Sex: Jenis kelamin (0 = Perempuan, 1 = Laki-laki)
- ChestPainType: Jenis nyeri dada
- RestingBP: Tekanan darah saat istirahat
- Cholesterol: Tingkat kolesterol
- FastingBS: Gula darah puasa (1 jika > 120 mg/dl, 0 jika tidak)
- RestingECG: Hasil EKG saat istirahat
- MaxHR: Detak jantung maksimum selama tes
- ExerciseAngina: Angina akibat latihan
- Oldpeak: Depresi ST
- ST_Slope: Kemiringan segmen ST
- HeartDisease: Target variabel (1 = mengidap penyakit jantung, 0 = tidak)

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.
### Menangani outliers
```
for column in num_cols:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr
    df[column] = df[column].clip(lb, ub)
```


## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.
Algoritma yang digunakan:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

