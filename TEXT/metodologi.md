<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Metodologi Prediksi Saham</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }
        h1, h2, h3 {
            color: #333;
        }
        ul {
            list-style-type: disc;
            margin: 0 0 1em 1.5em;
        }
        code {
            background: #f2f2f2;
            padding: 2px 4px;
            border-radius: 3px;
        }
        pre {
            background: #f2f2f2;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <h1><b>Metodologi Prediksi Saham</b></h1>
    <ol>
        <li><strong>Persiapan Lingkungan</strong>
            <p>Impor library yang diperlukan seperti TensorFlow, yfinance, numpy, pandas, dan lainnya.</p>
        </li>
        <li><strong>Pengumpulan Data</strong>
            <p>Data historis saham diambil menggunakan library <code>yfinance</code> dengan rentang data dari 1 Januari 2000 hingga hari ini.</p>
        </li>
        <li><strong>Pra-pemrosesan Data</strong>
            <p>Data di-scaling menggunakan <code>MinMaxScaler</code> dan diubah menjadi format yang sesuai untuk input ke model CNN-GRU.</p>
        </li>
        <li><strong>Perancangan Model CNN-GRU</strong>
            <p>Model CNN-GRU dirancang dengan arsitektur Sequential yang menggabungkan lapisan <code>Conv1D</code> untuk ekstraksi fitur, lapisan <code>GRU</code> untuk menangkap dependensi jangka panjang, <code>Dropout</code> untuk mencegah overfitting, dan <code>Dense</code> untuk menghasilkan output prediksi.</p>
        </li>
        <li><strong>Pelatihan Model</strong>
            <p>Model dilatih menggunakan data pelatihan dengan parameter seperti <code>epochs</code>, <code>batch_size</code>, dan <code>learning rate</code> yang disesuaikan untuk optimasi performa.</p>
        </li>
        <li><strong>Evaluasi Model</strong>
            <p>Model dievaluasi menggunakan metrik seperti MSE, RMSE, R2 Score, dan MAPE untuk mengukur akurasi prediksi.</p>
        </li>
        <li><strong>Visualisasi Prediksi dan Perhitungan Metrik</strong>
            <p>Hasil prediksi divisualisasikan bersama dengan data aktual, dan metrik performa model juga ditampilkan untuk analisis lebih lanjut.</p>
        </li>
        <li><strong>Interpretasi dan Pelaporan Hasil</strong>
            <p>Hasil prediksi diinterpretasikan dan dilaporkan dengan analisis tren pasar dan rekomendasi investasi yang disusun berdasarkan prediksi model.</p>
        </li>
    </ol>

    <h2>Diagram Alur Proses</h2>
    <pre>
    <code>
    graph TD
    A[Pengumpulan Data] --> B[Pemrosesan Data]
    B --> C[Pelatihan Model CNN-GRU]
    C --> D[Prediksi]
    D --> E[Visualisasi Hasil]
    D --> F[Evaluasi Model]
    F --> G[Cross-validation]
    G --> H[Perhitungan MAPE]
    H --> I[Kesimpulan]
    </code>
    </pre>

    <p>Metodologi ini memungkinkan prediksi yang akurat dengan mempertimbangkan berbagai faktor yang mempengaruhi harga saham, serta memberikan evaluasi yang komprehensif terhadap performa model dan menyajikan kesimpulan akhir.</p>
</body>
</html>
