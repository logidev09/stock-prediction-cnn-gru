<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gambaran Umum Aplikasi</title>
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
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1em;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>

<h1><b>Gambaran Umum Aplikasi</b></h1>
<p style="font-size: 1.1em; font-style: italic; color: #555; margin-bottom: 20px;">
    Portofolio ini dibuat oleh Ilham Rizkyansyah &middot; Universitas Gunadarma Informatika
</p>
<p><strong>Selamat datang</strong> di aplikasi prediksi saham bank-bank terkemuka di Indonesia. Aplikasi ini menyajikan analisis dan prediksi untuk saham dari lima bank terbesar di Indonesia:</p>
<ul>
    <li>PT Bank Central Asia Tbk (BCA)</li>
    <li>PT Bank Rakyat Indonesia Tbk (BRI)</li>
    <li>PT Bank Mandiri Tbk (Bank Mandiri)</li>
    <li>PT Bank Negara Indonesia Tbk (BNI)</li>
    <li>PT Bank Syariah Indonesia Tbk (BSI)</li>
</ul>

<h2>Fitur Utama:</h2>
<ul>
    <li>Informasi singkat tentang masing-masing bank</li>
	<li>Informasi Mengenai Script Code Library yang digunakan</li>
    <li>Data historis saham beserta Grafiknya secara Real-Time</li>
    <li>Memilih Jumlah Data Saham dengan Slider</li>
    <li>Menanpilkan Julah Data yang digunakan untuk Pelatihan dan Pengujian</li>
	<li>Informasi Mengenai Script Code Model yang digunakan</li>
    <li>Memilih beberapa hari untuk dilakukan prediksi dengan Fitur Multiselect</li>
	<li>Dapat Mengubah Jumlah Epoch dab Ukuran Batch Pelatihan</li>
    <li>Informasi Evaluasi Model Dengan MSE, RMSE, R2, MAPE, Accuracy, juga Grafik Loss dan Val Loss</li>
	<li>Menampilkan Hasil Prediksi dalam Bentuk Grafik, Tabel disertai dengan informasi performa perhitungan metrik</li>
	<li>Menampilkan Hasil Interpretasi dan Pelaporan Hasil: Harga Terakhir, Prediksi Harga, Tren Harga, Analisis Performa Model, Insight Pasar, Rekomendasi, Catatan Penting</li>
</ul>

<h2>Perbandingan Singkat:</h2>
<table>
    <tr>
        <th>Bank</th>
        <th>Kode Saham</th>
        <th>Fokus Utama</th>
        <th>Kepemilikan</th>
    </tr>
    <tr>
        <td>BCA</td>
        <td>BBCA.JK</td>
        <td>Ritel, UKM, Korporasi</td>
        <td>Swasta</td>
    </tr>
    <tr>
        <td>BRI</td>
        <td>BBRI.JK</td>
        <td>UMKM, Pertanian</td>
        <td>BUMN</td>
    </tr>
    <tr>
        <td>Mandiri</td>
        <td>BMRI.JK</td>
        <td>Korporasi, Komersial, Mikro & Ritel</td>
        <td>BUMN</td>
    </tr>
    <tr>
        <td>BNI</td>
        <td>BBNI.JK</td>
        <td>Korporasi, Ritel, Internasional</td>
        <td>BUMN</td>
    </tr>
    <tr>
        <td>BSI</td>
        <td>BRIS.JK</td>
        <td>Perbankan Syariah</td>
        <td>BUMN</td>
    </tr>
</table>

<p>Silakan pilih bank yang ingin Anda analisis di menu atas untuk memulai!</p>

<h2>Latar Belakang Aplikasi Prediksi Saham Bank</h2>
<p>Aplikasi ini dikembangkan dengan beberapa tujuan utama:</p>
<ol>
    <li><strong>Memudahkan Analisis Saham Perbankan:</strong> Menyediakan alat yang user-friendly untuk menganalisis dan memprediksi harga saham 5 bank terbesar di Indonesia.</li>
    <li><strong>Memanfaatkan Teknologi Machine Learning:</strong> Menggunakan model CNN-GRU dari Facebook untuk melakukan prediksi time series yang akurat.</li>
    <li><strong>Mendukung Pengambilan Keputusan Investasi:</strong> Membantu investor dan analis dalam membuat keputusan berdasarkan data historis dan prediksi.</li>
    <li><strong>Meningkatkan Literasi Keuangan:</strong> Memberikan pemahaman lebih baik tentang pergerakan harga saham dan faktor-faktor yang mempengaruhinya.</li>
    <li><strong>Menyediakan Visualisasi Data yang Komprehensif:</strong> Menampilkan grafik dan komponen prediksi untuk analisis yang lebih mendalam.</li>
    <li><strong>Fleksibilitas Periode Prediksi:</strong> Memungkinkan pengguna untuk memilih berbagai rentang waktu prediksi, dari 1 hari hingga 10 tahun.</li>
    <li><strong>Transparansi Model:</strong> Menyajikan metrik evaluasi model untuk memberikan gambaran tentang akurasi prediksi.</li>
</ol>

<p>Aplikasi ini bertujuan untuk menjembatani kesenjangan antara data keuangan yang kompleks dengan kebutuhan praktis para investor dan analis, sambil tetap menekankan pentingnya analisis tambahan dan konsultasi dengan ahli keuangan sebelum membuat keputusan investasi.</p>

</body>
</html>
