<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Glosarium Istilah Saham dan Prediksi</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }
        h1, h2 {
            color: #333;
        }
        ul {
            list-style-type: disc;
            margin: 0 0 1em 1.5em;
        }
    </style>
</head>
<body>
    <h1><b>Glosarium Istilah Saham dan Prediksi</b></h1>

    <h2>Istilah Umum Saham</h2>
    <ul>
        <li><strong>Saham</strong>: Bukti kepemilikan dalam sebuah perusahaan.</li>
        <li><strong>Bursa Efek</strong>: Pasar terorganisir untuk perdagangan efek (seperti saham).</li>
        <li><strong>Dividen</strong>: Bagian keuntungan perusahaan yang dibagikan kepada pemegang saham.</li>
        <li><strong>Volatilitas</strong>: Ukuran fluktuasi harga saham dalam periode tertentu.</li>
        <li><strong>Likuiditas</strong>: Kemudahan untuk membeli atau menjual saham tanpa menyebabkan perubahan harga yang signifikan.</li>
    </ul>

    <h2>Istilah Teknis Analisis</h2>
    <ul>
        <li><strong>Open</strong>: Harga pembukaan saham pada periode perdagangan tertentu.</li>
        <li><strong>Close</strong>: Harga penutupan saham pada periode perdagangan tertentu.</li>
        <li><strong>High</strong>: Harga tertinggi saham dalam periode perdagangan tertentu.</li>
        <li><strong>Low</strong>: Harga terendah saham dalam periode perdagangan tertentu.</li>
        <li><strong>Volume</strong>: Jumlah saham yang diperdagangkan dalam periode tertentu.</li>
    </ul>

    <h2>Metrik Prediksi</h2>
    <ul>
        <li><strong>MAPE (Mean Absolute Percentage Error)</strong>: Metrik yang mengukur akurasi prediksi dalam persentase.
            <br>Rumus: <code>MAPE = (1/n) * Σ|(Actual - Forecast) / Actual| * 100</code>
            <br>- n adalah jumlah periode
            <br>- Actual adalah nilai sebenarnya
            <br>- Forecast adalah nilai prediksi
        </li>
        <li><strong>MSE (Mean Squared Error)</strong>: Metrik yang mengukur rata-rata kuadrat kesalahan antara nilai aktual dan prediksi.
            <br>Rumus: <code>MSE = (1/n) * Σ(Actual - Forecast)^2</code>
        </li>
        <li><strong>RMSE (Root Mean Squared Error)</strong>: Akar kuadrat dari MSE, memberikan ukuran kesalahan dalam unit yang sama dengan data aslinya.
            <br>Rumus: <code>RMSE = √MSE</code>
        </li>
        <li><strong>R2 Score (Coefficient of Determination)</strong>: Metrik yang mengukur seberapa baik model menjelaskan variasi dalam data.
            <br>Rumus: <code>R2 = 1 - (Σ(Actual - Forecast)^2 / Σ(Actual - Mean(Actual))^2)</code>
        </li>
        <li><strong>Accuracy</strong>: Metrik yang mengukur seberapa sering model membuat prediksi yang benar.
            <br>Rumus: <code>Accuracy = (Number of Correct Predictions / Total Number of Predictions) * 100%</code>
        </li>
        <li><strong>Cross-validation</strong>: Teknik untuk menilai bagaimana hasil analisis statistik akan digeneralisasi ke set data independen.</li>
    </ul>

    <h2>Metodologi Pemodelan</h2>
    <ul>
        <li><strong>CNN-GRU</strong>: Model prediksi time series yang digunakan dalam dokumen ini.</li>
        <li><strong>Time Series</strong>: Serangkaian data poin yang diindeks secara berurutan seiring waktu.</li>
        <li><strong>Forecasting</strong>: Proses membuat prediksi masa depan berdasarkan data historis dan analisis tren.</li>
    </ul>

    <h2>Istilah Teknis Aplikasi</h2>
    <ul>
        <li><strong>yfinance</strong>: Library Python untuk mengunduh data keuangan dari Yahoo Finance, digunakan dalam analisis.</li>
        <li><strong>Plotly</strong>: Library visualisasi data interaktif yang digunakan untuk membuat grafik dan diagram.</li>
        <li><strong>Streamlit</strong>: Framework Python untuk membuat aplikasi web untuk analisis data dan machine learning.</li>
    </ul>

    <h2>Kode Saham</h2>
    <ul>
        <li><strong>BBCA.JK</strong>: Kode saham untuk PT Bank Central Asia Tbk di Bursa Efek Indonesia.</li>
        <li><strong>BBRI.JK</strong>: Kode saham untuk PT Bank Rakyat Indonesia Tbk di Bursa Efek Indonesia.</li>
        <li><strong>BMRI.JK</strong>: Kode saham untuk PT Bank Mandiri Tbk di Bursa Efek Indonesia.</li>
        <li><strong>BBNI.JK</strong>: Kode saham untuk PT Bank Negara Indonesia Tbk di Bursa Efek Indonesia.</li>
        <li><strong>BRIS.JK</strong>: Kode saham untuk PT Bank Syariah Indonesia Tbk di Bursa Efek Indonesia.</li>
    </ul>
</body>
</html>
