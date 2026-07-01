# Glosarium Istilah Saham dan Prediksi

## Istilah Umum Saham
* **Saham**: Bukti kepemilikan dalam sebuah perusahaan.
* **Bursa Efek**: Pasar terorganisir untuk perdagangan efek (seperti saham).
* **Dividen**: Bagian keuntungan perusahaan yang dibagikan kepada pemegang saham.
* **Volatilitas**: Ukuran fluktuasi harga saham dalam periode tertentu.
* **Likuiditas**: Kemudahan untuk membeli atau menjual saham tanpa menyebabkan perubahan harga yang signifikan.

## Istilah Teknis Analisis
* **Open**: Harga pembukaan saham pada periode perdagangan tertentu.
* **Close**: Harga penutupan saham pada periode perdagangan tertentu.
* **High**: Harga tertinggi saham dalam periode perdagangan tertentu.
* **Low**: Harga terendah saham dalam periode perdagangan tertentu.
* **Volume**: Jumlah saham yang diperdagangkan dalam periode tertentu.

## Metrik Prediksi
* **MAPE (Mean Absolute Percentage Error)**: Metrik yang mengukur akurasi prediksi dalam persentase.
  * **Rumus**: `MAPE = (1/n) * Σ|(Actual - Forecast) / Actual| * 100`
  * `n` adalah jumlah periode
  * `Actual` adalah nilai sebenarnya
  * `Forecast` adalah nilai prediksi
* **MSE (Mean Squared Error)**: Metrik yang mengukur rata-rata kuadrat kesalahan antara nilai aktual dan prediksi.
  * **Rumus**: `MSE = (1/n) * Σ(Actual - Forecast)^2`
* **RMSE (Root Mean Squared Error)**: Akar kuadrat dari MSE, memberikan ukuran kesalahan dalam unit yang sama dengan data aslinya.
  * **Rumus**: `RMSE = √MSE`
* **R2 Score (Coefficient of Determination)**: Metrik yang mengukur seberapa baik model menjelaskan variasi dalam data.
  * **Rumus**: `R2 = 1 - (Σ(Actual - Forecast)^2 / Σ(Actual - Mean(Actual))^2)`
* **Accuracy**: Metrik yang mengukur seberapa sering model membuat prediksi yang benar.
  * **Rumus**: `Accuracy = (Number of Correct Predictions / Total Number of Predictions) * 100%`
* **Cross-validation**: Teknik untuk menilai bagaimana hasil analisis statistik akan digeneralisasi ke set data independen.

## Metodologi Pemodelan
* **CNN-GRU**: Model prediksi time series yang menggabungkan Convolutional Neural Network (CNN) dan Gated Recurrent Unit (GRU).
* **Time Series**: Serangkaian data poin yang diindeks secara berurutan seiring waktu.
* **Forecasting**: Proses membuat prediksi masa depan berdasarkan data historis dan analisis tren.

## Istilah Teknis Aplikasi
* **yfinance**: Library Python untuk mengunduh data keuangan dari Yahoo Finance.
* **Plotly**: Library visualisasi data interaktif.
* **Streamlit**: Framework Python untuk membuat aplikasi web interaktif.

## Kode Saham
* **BBCA.JK**: Kode saham PT Bank Central Asia Tbk.
* **BBRI.JK**: Kode saham PT Bank Rakyat Indonesia Tbk.
* **BMRI.JK**: Kode saham PT Bank Mandiri Tbk.
* **BBNI.JK**: Kode saham PT Bank Negara Indonesia Tbk.
* **BRIS.JK**: Kode saham PT Bank Syariah Indonesia Tbk.
