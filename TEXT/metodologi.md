# Metodologi Prediksi Saham

1. **Persiapan Lingkungan**
   Impor library yang diperlukan seperti TensorFlow, yfinance, numpy, pandas, dan lainnya.
   
2. **Pengumpulan Data**
   Data historis saham diambil menggunakan library `yfinance` dengan rentang data dari 1 Januari 2000 hingga hari ini.
   
3. **Pra-pemrosesan Data**
   Data di-scaling menggunakan `MinMaxScaler` dan diubah menjadi format yang sesuai untuk input ke model CNN-GRU.
   
4. **Perancangan Model CNN-GRU**
   Model CNN-GRU dirancang dengan arsitektur Sequential yang menggabungkan lapisan `Conv1D` untuk ekstraksi fitur, lapisan `GRU` untuk menangkap dependensi jangka panjang, `Dropout` untuk mencegah overfitting, dan `Dense` untuk menghasilkan output prediksi.
   
5. **Pelatihan Model**
   Model dilatih menggunakan data pelatihan dengan parameter seperti `epochs`, `batch_size`, dan `learning rate` yang disesuaikan untuk optimasi performa.
   
6. **Evaluasi Model**
   Model dievaluasi menggunakan metrik seperti MSE, RMSE, R2 Score, dan MAPE untuk mengukur akurasi prediksi.
   
7. **Visualisasi Prediksi dan Perhitungan Metrik**
   Hasil prediksi divisualisasikan bersama dengan data aktual, dan metrik performa model juga ditampilkan untuk analisis lebih lanjut.
   
8. **Interpretasi dan Pelaporan Hasil**
   Hasil prediksi diinterpretasikan dan dilaporkan dengan analisis tren pasar dan rekomendasi investasi yang disusun berdasarkan prediksi model.

---

## Diagram Alur Proses

<div style="font-family: Arial, sans-serif; margin: 20px 0; padding: 20px; background-color: #f9f9f9; border-radius: 10px; border: 1px solid #eee;">
    <div style="display: flex; flex-direction: column; align-items: center;">
        <div style="background-color: #003A70; color: white; padding: 12px 24px; border-radius: 8px; margin: 8px; font-weight: bold; width: 260px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">1. Pengumpulan Data</div>
        <div style="font-size: 20px; color: #777;">↓</div>
        <div style="background-color: #00529C; color: white; padding: 12px 24px; border-radius: 8px; margin: 8px; font-weight: bold; width: 260px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">2. Pemrosesan Data</div>
        <div style="font-size: 20px; color: #777;">↓</div>
        <div style="background-color: #0060AF; color: white; padding: 12px 24px; border-radius: 8px; margin: 8px; font-weight: bold; width: 260px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">3. Pelatihan Model CNN-GRU</div>
        <div style="font-size: 20px; color: #777;">↓</div>
        <div style="background-color: #006885; color: white; padding: 12px 24px; border-radius: 8px; margin: 8px; font-weight: bold; width: 260px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">4. Prediksi</div>
        <div style="font-size: 20px; color: #777;">↓</div>
        <div style="display: flex; justify-content: space-between; width: 100%; max-width: 550px; margin: 15px 0;">
            <div style="display: flex; flex-direction: column; align-items: center; width: 46%;">
                <div style="background-color: #107EDE; color: white; padding: 12px; border-radius: 8px; font-weight: bold; width: 100%; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">5a. Visualisasi Hasil</div>
            </div>
            <div style="display: flex; flex-direction: column; align-items: center; width: 46%;">
                <div style="background-color: #00A39D; color: white; padding: 12px; border-radius: 8px; font-weight: bold; width: 100%; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">5b. Evaluasi Model</div>
                <div style="font-size: 20px; color: #777; margin: 6px 0;">↓</div>
                <div style="background-color: #2E7D32; color: white; padding: 12px; border-radius: 8px; font-weight: bold; width: 100%; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">6. Cross-validation</div>
                <div style="font-size: 20px; color: #777; margin: 6px 0;">↓</div>
                <div style="background-color: #D84315; color: white; padding: 12px; border-radius: 8px; font-weight: bold; width: 100%; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">7. Perhitungan MAPE</div>
                <div style="font-size: 20px; color: #777; margin: 6px 0;">↓</div>
                <div style="background-color: #37474F; color: white; padding: 12px; border-radius: 8px; font-weight: bold; width: 100%; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">8. Kesimpulan</div>
            </div>
        </div>
    </div>
</div>

Metodologi ini memungkinkan prediksi yang akurat dengan mempertimbangkan berbagai faktor yang mempengaruhi harga saham, serta memberikan evaluasi yang komprehensif terhadap performa model dan menyajikan kesimpulan akhir.
