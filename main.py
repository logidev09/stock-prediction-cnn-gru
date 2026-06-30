import time
import numpy as np
import pandas as pd
import yfinance as yf
try:
    import crypto_yfinance as cyf
except ImportError:
    crypto_yfinance = None
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
from tensorflow.keras.optimizers import Adam
from streamlit_option_menu import option_menu
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import Conv1D, GRU, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

def main(stock):
    st.header(f"Prediksi Harga Saham dengan kode {stock}")

    with st.expander("1. Persiapan Lingkungan"):
        with st.spinner("Mengimpor library yang diperlukan..."):

            code = '''import time
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
from tensorflow.keras.optimizers import Adam
from streamlit_option_menu import option_menu
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import Conv1D, GRU, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score'''

            lines = code.split('\n')
            placeholder = st.empty()

            for i in range(len(lines) + 1):
                placeholder.code('\n'.join(lines[:i]), language='python')
                time.sleep(0.1)  # Adjust this value to control the speed of the animation

            st.success("Library berhasil diimpor")

    with st.expander("2. Pengumpulan Data"):

        #Menyimpan Data pada Cache
        @st.cache_data
        def load_data(ticker, start_date, end_date):
            try:
                # Try crypto first if it looks like a crypto symbol
                if crypto_yfinance and (ticker.endswith('-USD') or ticker.endswith('-IDR') or ticker in ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'DOT-USD', 'SHIB-USD', 'AVAX-USD']):
                    data = cyf.download(ticker, start=start_date, end=end_date)
                else:
                    # Try as regular stock
                    data = yf.download(ticker, start=start_date, end=end_date)
                
                if data.empty:
                    st.error(f"Tidak dapat memuat data untuk {ticker}. Silakan coba simbol lain.")
                    return pd.DataFrame()
                    
                data.reset_index(inplace=True)
                
                # Ensure we have a Date column
                if 'Date' not in data.columns and 'Datetime' in data.columns:
                    data.rename(columns={'Datetime': 'Date'}, inplace=True)
                elif 'Date' not in data.columns and data.index.name == 'Date':
                    data['Date'] = data.index
                    data.reset_index(drop=True, inplace=True)
                elif 'Date' not in data.columns:
                    # If no date column exists, create one from index
                    data['Date'] = data.index
                    data.reset_index(drop=True, inplace=True)
                    
                return data
            except Exception as e:
                st.error(f"Error loading data for {ticker}: {str(e)}")
                return pd.DataFrame()

        # DATA HISTORY
        full_data = load_data(stock, "2000-01-01", date.today().strftime("%Y-%m-%d"))

        st.subheader("Data keseluruhan")
        st.write(full_data.head(1))
        st.write("Hingga")
        st.write(full_data.tail(1))

        # Mengubah index menjadi datetime untuk memudahkan plotting
        // Handle Date column - ensure it exists and is datetime
        if 'Date' not in full_data.columns:
            if full_data.index.name == 'Date' or full_data.index.name == 'Datetime':
                full_data.reset_index(inplace=True)
                if 'Datetime' in full_data.columns:
                    full_data.rename(columns={'Datetime': 'Date'}, inplace=True)
            else:
                full_data['Date'] = full_data.index
                full_data.reset_index(drop=True, inplace=True)
        
        # Ensure Date is datetime
        if 'Date' in full_data.columns:
            full_data['Date'] = pd.to_datetime(full_data['Date'])
            full_data.set_index('Date', inplace=True)

        # Membuat chart dengan matplotlib untuk data keseluruhan
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        ax1.plot(full_data.index, full_data['Close'], label='Harga Saham', color='#31333F')
        ax1.set_title('Data Keseluruhan Harga Saham')
        ax1.set_xlabel('Tanggal')
        ax1.set_ylabel('Harga Saham')
        ax1.legend()
        
        # Format x-axis
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
        
        plt.tight_layout()
        st.pyplot(fig1)

        with st.popover("Tampilkan Semua Data"):
            st.write(full_data)

        # DATA PELATIHAN
        # Pilihan untuk input jumlah data pelatihan

        if stock == "BBCA.JK" or stock == "BMRI.JK" or stock == "BBNI.JK":
            use_days = st.checkbox("Gunakan jumlah hari terakhir")
            if use_days:
                days = st.number_input("Jumlah hari untuk pelatihan", min_value=120, max_value=360*24, value=1800)

            else:
                # Membuat Slider untuk memilih data Pelatihan
                years_ago = st.slider('Pilih berapa tahun yang lalu untuk pelatihan:', 0, 24, 5)
                months_ago = st.slider('Pilih berapa bulan tambahan yang lalu untuk pelatihan:', 0, 11, 0)

                # Menghitung total bulan dan tanggal mulai
                total_months = years_ago * 12 + months_ago
                days = total_months*30

            with st.popover("Tips Memilih Data Pelatihan"):
                st.info('Ket: Semakin lama hari yang dipilih, maka jumlah hari Prediksi dapat dilakukan dengan lebih banyak, namun prediksi menjadi lebih tidak akurat.', icon=":material/notes:")
                st.warning('Ket: Secara Default menggunakan 5 Tahun atau 1800 hari yang lalu, yang hanya dapat melakukan prediksi hingga 7 Bulan kedepan.', icon=":material/pan_tool_alt:")
                st.warning('Ket: Jumlah Minimal 4 Bulan atau 120 hari yang lalu, yang hanya dapat melakukan prediksi hingga 3 hari kedepan, Perhatikanlah pada bagian Pra-pemrosesan data "Ukuran data pengujian" jumlahnya sebanding dengan jumlah hari yang dapat anda lakukan untuk prediksi kedepan.', icon=":material/exclamation:")

        else:
            use_days = st.checkbox("Gunakan jumlah hari terakhir")
            if use_days:
                days = st.number_input("Jumlah hari untuk pelatihan", min_value=120, max_value=360*24, value=1470)

            else:
                # Membuat Slider untuk memilih data Pelatihan
                years_ago = st.slider('Pilih berapa tahun yang lalu untuk pelatihan:', 0, 24, 4)
                months_ago = st.slider('Pilih berapa bulan tambahan yang lalu untuk pelatihan:', 0, 11, 1)

                # Menghitung total bulan dan tanggal mulai
                total_months = years_ago * 12 + months_ago
                days = total_months*30

            with st.popover("Tips Memilih Data Pelatihan"):
                st.info('Ket: Semakin lama hari yang dipilih, maka jumlah hari Prediksi dapat dilakukan dengan lebih banyak, namun prediksi menjadi lebih tidak akurat.', icon=":material/notes:")
                st.warning('Ket: Secara Default menggunakan 4 Tahun 1 Bulan atau 1470 hari yang lalu, yang hanya dapat melakukan prediksi hingga 6 Bulan kedepan.', icon=":material/pan_tool_alt:")
                st.warning('Ket: Jumlah Minimal 4 Bulan atau 120 hari yang lalu, yang hanya dapat melakukan prediksi hingga 3 hari kedepan, Perhatikanlah pada bagian Pra-pemrosesan data "Ukuran data pengujian" jumlahnya sebanding dengan jumlah hari yang dapat anda lakukan untuk prediksi kedepan.', icon=":material/exclamation:")

        # Menjadikan hari terkahir adalah hari ini
        end_date = date.today()

        # Mengubah Format Tanggal data Pelatihan
        start_date = end_date - timedelta(days)
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

        st.write(f"Jumlah Hari yang dipilih **{days}**.")

        # Load data sesuai dengan rentang yang dipilih
        @st.cache_data
        def load_training_data(stock, start_date, end_date):
            return load_data(stock, start_date, end_date)

        # Fitur Beta
        try:
            data = load_training_data(stock, start_date, end_date) # yang sebelumnya hanya ini
        except Exception as e:
            print(f"Error loading data for {stock}: {e}")
            data = load_training_data(stock, start_date, end_date)
        # hingga ini

        st.subheader("Data Pelatihan yang telah dipilih")
        st.write(data.head(1))
        st.write("Hingga")
        st.write(data.tail(1))

        # Mengubah index menjadi datetime untuk data pelatihan
        // Handle Date column for training data
        if 'Date' not in data.columns:
            if data.index.name == 'Date' or data.index.name == 'Datetime':
                data.reset_index(inplace=True)
                if 'Datetime' in data.columns:
                    data.rename(columns={'Datetime': 'Date'}, inplace=True)
            else:
                data['Date'] = data.index
                data.reset_index(drop=True, inplace=True)
        
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)

        # Membuat chart dengan matplotlib untuk data pelatihan
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        ax2.plot(data.index, data['Close'], label='Harga Saham', color='#d6c36b')
        ax2.set_title('Data Pelatihan Harga Saham')
        ax2.set_xlabel('Tanggal')
        ax2.set_ylabel('Harga Saham')
        ax2.legend()
        
        # Format x-axis
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
        
        plt.tight_layout()
        st.pyplot(fig2)

        with st.popover("Tampilkan Semua Data Pelatihan"):
            st.write(data)

    with st.expander("3. Pra-pemrosesan Data"):

        if days >= 120:

            def preprocess_data(data, seq_length):
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

                X, y = [], []
                for i in range(seq_length, len(scaled_data)):
                    X.append(scaled_data[i-seq_length:i, 0])
                    y.append(scaled_data[i, 0])

                X, y = np.array(X), np.array(y)

                split = int(0.8 * len(X))
                x_train, x_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
                x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

                return x_train, x_test, y_train, y_test, scaler

            seq_length = 60
            x_train, x_test, y_train, y_test, scaler = preprocess_data(data, seq_length)

            # Menghitung persentase data pelatihan dan pengujian
            total_samples = x_train.shape[0] + x_test.shape[0]
            train_percentage = (x_train.shape[0] / total_samples) * 100
            test_percentage = (x_test.shape[0] / total_samples) * 100

            st.code('''seq_length = 60''')

            st.code('''x_train, x_test, y_train, y_test, scaler = preprocess_data(data, seq_length)''')

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ukuran data pelatihan", f"{x_train.shape[0]} sampel")
                st.metric("Persentase data pelatihan", f"{train_percentage:.2f}%")
            with col2:
                st.metric("Ukuran data pengujian", f"{x_test.shape[0]} sampel")
                st.metric("Persentase data pelatihan", f"{test_percentage:.2f}%")

            st.success("Pra-pemrosesan Data selesai!")
        else:
            st.warning('Harus Memilih Jumlah Hari Minimal 4 Bulan atau 120 hari', icon=":material/exclamation:")

    with st.expander("4. Perancangan Model CNN-GRU"):

        if days >= 120:

            st.subheader("Arsitektur Model:")

            def create_model(seq_length):
                model = Sequential([
                    Conv1D(filters=64,
                        kernel_size=3,
                        activation='relu',
                        input_shape=(seq_length, 1)),
                    GRU(50, return_sequences=True),
                    Dropout(0.2),
                    GRU(50),
                    Dense(1)
                ])

                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                return model

            def get_model(seq_length):
                return create_model(seq_length)

            code = '''def create_model(seq_length):
    model = Sequential([
        Conv1D(filters=64,
            kernel_size=3,
            activation='relu',
            input_shape=(seq_length, 1)),
        GRU(50, return_sequences=True),
        Dropout(0.2),
        GRU(50),
        Dense(1)
    ])'''
            lines = code.split('\n')
            placeholder = st.empty()

            for i in range(len(lines) + 1):
                placeholder.code('\n'.join(lines[:i]), language='python')
                time.sleep(0.1)  # Adjust this value to control the speed of the animation

            st.code('''  model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model''')

            st.code('''def get_model(seq_length):
    return create_model(seq_length)''')

            st.success("Perancangan Model CNN-GRU selesai!")

        else:
            st.warning('Harus Memilih Jumlah Hari Minimal 4 Bulan atau 120 hari', icon=":material/exclamation:")

    with st.expander("5. Pelatihan Model", True):

        # Menambahkan nilai default untuk ketika tombol belum ditekan
        btn_check = 0

        if days >= 120:

            with st.popover("Mengubah Jumlah Epoch"):
                epochs = st.select_slider("Jumlah Epoch", options=[1] + list(range(10, 101, 10)), value=40)
                st.info('Ket: Semakin banyak Jumlahnya, maka semakin lambat waktu untuk komputasinya.', icon=":material/thumb_down:")
                st.warning('Ket: Semakin sedikit Jumlahnya, maka semakin cepat waktu untuk komputasinya, Namun mengurangi Performa Akurasi.', icon=":material/timer_3_alt_1:")

            with st.popover("Mengubah Ukuran Batch"):
                batch_size_options = [4, 8, 16, 32, 64, 128, 256]
                batch_size = st.select_slider("Ukuran Batch", options=batch_size_options, value=16)
                st.info('Ket: Semakin besar ukurannya, maka semakin cepat waktu untuk komputasinya.', icon=":material/thumb_up:")
                st.warning('Ket: Semakin kecil ukurannya, maka semakin lambat waktu untuk komputasinya, Namun meningkatkan Performa Akurasi.', icon=":material/timer_10_alt_1:")

            # Define the time periods and their corresponding days
            def get_forecast_options(stock):
                forecast_options = [
                    ("1 Hari", 1), ("2 Hari", 2), ("3 Hari", 3), ("4 Hari", 4), ("5 Hari", 5), ("6 Hari", 6),
                    ("1 Minggu", 7), ("2 Minggu", 14), ("3 Minggu", 21), ("1 Bulan", 30), ("2 Bulan", 60),
                    ("3 Bulan", 90), ("4 Bulan", 120), ("5 Bulan", 150), ("6 Bulan", 180), ("7 Bulan", 210),
                    ("8 Bulan", 240), ("9 Bulan", 270), ("10 Bulan", 300), ("11 Bulan", 330), ("1 Tahun", 365),
                    ("2 Tahun", 730)
                ]
                return forecast_options

            # Define the default options for each time period
            default_options_map = {
                3: ["1 Hari", "2 Hari", "3 Hari"],
                4: ["2 Hari", "3 Hari", "4 Hari"],
                5: ["3 Hari", "4 Hari", "5 Hari"],
                6: ["4 Hari", "5 Hari", "6 Hari"],
                7: ["5 Hari", "6 Hari", "1 Minggu"],
                14: ["6 Hari", "1 Minggu", "2 Minggu"],
                21: ["1 Minggu", "2 Minggu", "3 Minggu"],
                30: ["1 Minggu", "2 Minggu", "1 Bulan"],
                60: ["1 Minggu", "1 Bulan", "2 Bulan"],
                90: ["1 Minggu", "1 Bulan", "3 Bulan"],
                120: ["1 Minggu", "1 Bulan", "3 Bulan", "4 Bulan"],
                150: ["1 Minggu", "1 Bulan", "3 Bulan", "5 Bulan"],
                180: ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan"],
                210: ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan", "7 Bulan"],
                240: ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan", "8 Bulan"],
                270: ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan", "9 Bulan"],
                300: ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan", "10 Bulan"],
                330: ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan", "11 Bulan"],
                365: ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan", "1 Tahun"],
                730: ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan", "1 Tahun", "2 Tahun"]
            }

            def initialize_forecast_options(stock, x_test):
                forecast_options = get_forecast_options(stock)
                forecast_options_dict = {name: days for name, days in forecast_options}
                forecast_days = x_test.shape[0]

                # Filter opsi forecast yang tidak melebihi jumlah hari dalam x_test
                valid_forecast_options = {name: days for name, days in forecast_options_dict.items() if days <= forecast_days}

                # Jika tidak ada opsi yang valid, gunakan opsi terpanjang yang tersedia
                if not valid_forecast_options:
                    max_valid_days = max(days for days in forecast_options_dict.values() if days <= forecast_days)
                    valid_forecast_options = {name: days for name, days in forecast_options_dict.items() if days == max_valid_days}

                # Pilih opsi default berdasarkan kecocokan terdekat dengan forecast_days
                closest_key = min(default_options_map.keys(), key=lambda x: abs(x - forecast_days))
                default_options = default_options_map[closest_key]

                # Pastikan semua opsi default valid untuk stok saat ini dan tidak melebihi forecast_days
                default_options = [option for option in default_options if option in valid_forecast_options]

                return valid_forecast_options, default_options

            # Penggunaan fungsi
            forecast_options_dict, default_options = initialize_forecast_options(stock, x_test)

            # Streamlit UI untuk memilih periode forecast
            selected_periods = st.multiselect(
                "Pilih Periode Forecasting",
                options=list(forecast_options_dict.keys()),
                default=default_options
            )

            # DATA PELATIHAN
            end_date = date.today()

            # Pastikan end_date adalah objek datetime
            if isinstance(end_date, str):
                end_date = date.strptime(end_date, "%Y-%m-%d")

            # Cache the training function
            def train_model(x_train, y_train, epochs, batch_size, _on_epoch_end):
                with st.spinner('Sedang Melatih model... Harap tunggu.'):
                    try:
                        model = get_model(x_train.shape[1])
                        history = model.fit(
                            x_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.1,
                            verbose=0,
                            callbacks=[LambdaCallback(on_epoch_end=_on_epoch_end)]
                        )
                        return model, history
                    except Exception as e:
                        st.error(f"Terjadi error saat melatih model: {str(e)}")
                        st.error('Silakan coba lagi dengan parameter yang berbeda.', icon=":material/pan_tool_alt:")
                        return None, None

            if st.button("Latih Model", type="primary"):
                start_time = time.time()
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_estimate = st.empty()

                def on_epoch_end(epoch, logs):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch + 1}/{epochs}")

                    elapsed_time = time.time() - start_time
                    estimated_total_time = elapsed_time / progress
                    remaining_time = estimated_total_time - elapsed_time
                    time_estimate.text(f"Estimasi waktu tersisa: {remaining_time:.2f} detik")

                model, history = train_model(x_train, y_train, epochs, batch_size, on_epoch_end)

                if history:
                    end_time = time.time()
                    training_time = end_time - start_time
                    st.success(f"Pelatihan Model selesai! Waktu komputasi total: {training_time:.2f} detik")

                # Menambahkan nilai default untuk ketika tombol sudah ditekan
                btn_check = 1
        else:
            st.warning('Harus Memilih Jumlah Hari Minimal 4 Bulan atau 120 hari', icon=":material/exclamation:")

    with st.expander("6. Evaluasi Model"):

        if btn_check == 1:

            with st.spinner('Mengevaluasi model... Harap tunggu.'):

                y_pred = model.predict(x_test)
                y_pred = scaler.inverse_transform(y_pred).flatten()
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                actual_dates = data.index[-len(y_test):]

                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)

                st.subheader("Metrik Evaluasi:")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MSE", f"{mse:.3f}")
                    st.metric("RMSE", f"{rmse:.3f}")
                    st.metric("R2 Score", f"{r2:.3f}")
                    st.metric("MAPE", f"{mape:.3f}")
                    st.metric("Akurasi", f"{100 - mape*100:.3f}%")

                with col2:
                    # Menampilkan tabel perbandingan
                    comparison_df = pd.DataFrame({
                        'Tanggal': actual_dates.strftime('%Y-%m-%d'),
                        'Harga Aktual': y_test.round(2),
                        'Harga Prediksi': y_pred.round(2)
                    })
                    st.dataframe(comparison_df)

                if isinstance(rmse, (int, float)):
                    if rmse < 50:
                        st.success('Performa: Sangat Baik', icon=":material/thumb_up:")
                    elif rmse < 90:
                        st.success('Performa: Baik', icon=":material/thumb_up:")
                    elif rmse < 130:
                        st.info('Performa: Cukup Baik', icon=":material/thumb_up:")
                    elif rmse < 170:
                        st.info('Performa: Kurang Baik', icon=":material/thumb_down:")
                    elif rmse < 210:
                        st.error('Performa: Buruk', icon=":material/thumb_down:")
                    else:
                        st.error('Performa: Sangat Buruk', icon=":material/thumb_down:")
                else:
                    st.error(f"Unexpected type for rmse: {type(rmse)}")

                # Menampilkan Plot
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(actual_dates, y_test, label='Harga Aktual', color='#D6C36B')
                ax.plot(actual_dates, y_pred, label='Harga Pengujian', color='#B16ED0')

                st.subheader("Visualisasi Hasil")
                ax.set_title('Perbandingan Harga Aktual dan Prediksi')
                ax.set_xlabel('Tanggal')
                ax.set_ylabel('Harga Saham')
                ax.legend()

                # Format x-axis
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))

                plt.tight_layout()
                st.pyplot(fig)

                with st.popover("Menampilkan Grafik Loss dan Val Loss"):
                        # Display final metrics
                        final_loss = history.history['loss'][-1]
                        final_val_loss = history.history['val_loss'][-1]
                        st.metric("Loss akhir", f"{final_loss:.4f}")
                        st.metric("Validation Loss akhir", f"{final_val_loss:.4f}")

                        # Display full training history
                        st.subheader("Riwayat Pelatihan")
                        st.line_chart(pd.DataFrame(history.history))

                st.success("Evaluasi Model selesai!")
        else:
            st.warning('Harus Melakukan Pelatihan Model Terlebih dahulu', icon=":material/exclamation:")

    with st.expander("7. Visualisasi Prediksi dan Perhitungan Metrik"):

        def forecast_future(model, last_sequence, scaler, n_steps):

            forecast = []

            current_sequence = last_sequence.copy()

            for _ in range(n_steps):
                prediction = model.predict(current_sequence.reshape(1, current_sequence.shape[0], 1))
                forecast.append(prediction[0, 0])
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = prediction

            forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

            return forecast

        if btn_check == 1:

            if mape < 0.1:

                start_time = time.time()
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_estimate = st.empty()

                with st.spinner('Sedang Melakukan prediksi dan perhitungan metrik... Harap tunggu.'):

                    for i, forecast_period in enumerate(selected_periods):

                        forecast_days = forecast_options_dict[forecast_period]
                        last_sequence = x_test[-1]
                        forecast = forecast_future(model, last_sequence, scaler, forecast_days)

                        progress = (i + 1) / len(selected_periods)
                        progress_bar.progress(progress)

                        elapsed_time = time.time() - start_time
                        estimated_total_time = elapsed_time / progress
                        remaining_time = estimated_total_time - elapsed_time

                        # Menghitung Estimasi waktu
                        time_estimate.text(f"Estimasi waktu tersisa: {remaining_time:.2f} detik")
                        last_date = data.index[-1]
                        date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

                        st.subheader(f"Prediksi untuk {forecast_period}:")
                        fig, ax = plt.subplots(figsize=(10, 6))

                        # Determine the appropriate start index for plotting
                        start_idx = -(forecast_days*3)

                        ax.plot(data.index[start_idx:], data['Close'].values[start_idx:], label='Harga Aktual', color='#D6C36B')
                        ax.plot(actual_dates[start_idx:], y_pred[start_idx:], label='Harga Pengujian', color='#B16ED0')
                        ax.plot(date_range, forecast, label='Harga Prediksi', color='#107EDE')
                        ax.set_xlabel('Tanggal')
                        ax.set_ylabel('Harga Saham')
                        ax.legend()

                        # Format x-axis
                        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
                        plt.tight_layout()
                        st.pyplot(fig)

                        # Data Line untuk grafik
                        last_actual_price = float(data['Close'].iloc[-1]) if isinstance(data['Close'].iloc[-1], (int, float)) else float(data['Close'].iloc[-1].item())  # Extract scalar value
                        last_forecast_price = float(forecast[-1][0]) if isinstance(forecast[-1][0], (int, float)) else float(forecast[-1][0].item())  # Extract scalar value
                        percent_change = ((last_forecast_price - last_actual_price) / last_actual_price) * 100

                        if len(y_test) >= forecast_days:

                            st.subheader("Tabel dan Metrik Performa:")

                            table_df = pd.DataFrame({
                                        'Tanggal': date_range.strftime('%Y-%b-%d'),
                                        'Harga Prediksi': forecast.flatten().round(2)
                                    })

                            # Calculate metrics
                            mse = mean_squared_error(y_test[:forecast_days], y_pred[:forecast_days])
                            rmse = np.sqrt(mse)
                            r2 = r2_score(y_test[:forecast_days], y_pred[:forecast_days])
                            mape = mean_absolute_percentage_error(y_test[:forecast_days], y_pred[:forecast_days])
                            accuracy = 100 - mape * 100

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("MSE", f"{mse:.3f}")
                                st.metric("MAPE", f"{mape:.3f}")
                                # Menampilkan tabel perbandingan
                                with st.popover("Tampilkan Tabel"):
                                    st.dataframe(table_df, width="stretch")
                            with col2:
                                st.metric("RMSE", f"{rmse:.3f}")
                                st.metric("R2 Score", f"{r2:.3f}")

                                st.metric("Akurasi", f"{accuracy:.3f}%")

                            st.subheader("Ringkasan Prediksi")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Harga Terakhir", f"Rp {last_actual_price:.2f}")
                            with col2:
                                st.metric("Prediksi Harga", f"Rp {last_forecast_price:.2f}", f"{percent_change:.2f}%")

                            if rmse < 50:
                                st.success('Performa: Sangat Baik', icon=":material/thumb_up:")
                            elif rmse < 90:
                                st.success('Performa: Baik', icon=":material/thumb_up:")
                            elif rmse < 130:
                                st.info('Performa: Cukup Baik', icon=":material/thumb_up:")
                            elif rmse < 170:
                                st.info('Performa: Kurang Baik', icon=":material/thumb_down:")
                            elif rmse < 210:
                                st.error('Performa: Buruk', icon=":material/thumb_down:")
                            else:
                                st.error('Performa: Sangat Buruk', icon=":material/thumb_down:")

                        else:
                            st.warning(f"Data tidak cukup untuk periode {forecast_period}, silahkan atur kembali jumlah hari pelatihan pada 'Pengumpulan data'.", icon=":material/exclamation:")

                        st.write("---")

                end_time = time.time()

                st.success(f"Prediksi dan perhitungan metrik selesai! Waktu komputasi total: {end_time - start_time:.2f} detik")

            else:
                st.warning("Silahkan pilih epoch yang lebih besar atau mengggunakan Jumlah Data Pelatihan yang lebih banyak, lalu lakukan kembali Pelatihan Model", icon=":material/exclamation:")

        else:
            st.warning('Harus Melakukan Pelatihan Model Terlebih dahulu', icon=":material/exclamation:")

    with st.expander("9. Interpretasi dan Pelaporan Hasil", True):

        # Mengecek apakah sudah menekan tombol Latih Model
        if btn_check == 1:

            if mape < 0.1:
                
                if 'last_actual_price' in locals() and 'last_forecast_price' in locals() and 'percent_change' in locals():

                    st.subheader(f"Ringkasan Prediksi **{forecast_period}** ke depan")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Harga Terakhir", f"Rp {last_actual_price:.2f}")
                    with col2:
                        st.metric("Prediksi Harga", f"Rp {last_forecast_price:.2f}", f"{percent_change:.2f}%")

                    def interpret_forecast(percent_change):
                        if percent_change < -20:
                            return "Tren harga saham diprediksi akan sangat turun 🔴."
                        elif percent_change < -5:
                            return "Tren harga saham diprediksi akan turun 🟠."
                        elif percent_change < 5:
                            return "Harga saham diprediksi akan stabil ⚫."
                        elif percent_change < 20:
                            return "Tren harga saham diprediksi akan naik 🟡."
                        else:
                            return "Tren harga saham diprediksi akan sangat naik 🟢."

                    interpretation = interpret_forecast(percent_change)

                    st.write(interpretation)

                    if rmse < 50:
                        st.success('Performa: Sangat Baik', icon=":material/thumb_up:")
                    elif rmse < 90:
                        st.success('Performa: Baik', icon=":material/thumb_up:")
                    elif rmse < 130:
                        st.info('Performa: Cukup Baik', icon=":material/thumb_up:")
                    elif rmse < 170:
                        st.info('Performa: Kurang Baik', icon=":material/thumb_down:")
                    elif rmse < 210:
                        st.error('Performa: Buruk', icon=":material/thumb_down:")
                    else:
                        st.error('Performa: Sangat Buruk', icon=":material/thumb_down:")

                    # Fungsi tambahan untuk analisis dan rekomendasi
                    def analyze_market_trends(data, forecast):
                        # Implementasi analisis tren pasar
                        # Contoh sederhana:
                        recent_trend = "bullish" if data['Close'].pct_change().mean().item() > 0 else "bearish"
                        forecast_trend = "naik" if forecast[-1] > forecast[0] else "turun"
                        return f"Tren pasar terkini cenderung {recent_trend}. Berdasarkan prediksi, harga saham diperkirakan akan {forecast_trend} dalam periode mendatang."

                    def generate_recommendation(percent_change, accuracy):
                        if accuracy > 80:
                            return "Prediksi menunjukkan penurunan yang signifikan dengan tingkat akurasi yang tinggi. Waspadai risiko dan pertimbangkan untuk mengurangi eksposur atau melakukan hedging."
                        else:
                            return "Prediksi menunjukkan pergerakan moderat. Pantau perkembangan pasar dan lakukan analisis lebih lanjut sebelum mengambil keputusan."
                else:
                    return
                        
                st.subheader("Insight Pasar")
                market_trends = analyze_market_trends(data, forecast)
                st.write(market_trends)
                
                st.subheader("Rekomendasi")
                recommendation = generate_recommendation(percent_change, accuracy)
                st.write(recommendation)
                
                st.warning('Catatan Penting', icon=":material/edit_note:")
                st.write("- Prediksi ini didasarkan pada data historis dan model statistik.")
                st.write("- Faktor eksternal seperti kondisi ekonomi, kebijakan perusahaan, dan peristiwa global dapat mempengaruhi harga saham secara signifikan.")
                st.write("- Selalu lakukan analisis tambahan dan konsultasikan dengan penasihat keuangan sebelum membuat keputusan investasi.")
            
            else:
                st.warning('Silahkan pilih epoch yang lebih besar atau menggunakan Jumlah Data Pelatihan yang lebih banyak, lalu lakukan kembali Pelatihan Model', icon=":material/exclamation:")
        else:
            st.warning('Harus Melakukan Pelatihan Model Terlebih dahulu', icon=":material/exclamation:")

# Pengguna memilih bank yang ingin dianalisis dari sidebar Streamlit. Pilihan bank termasuk BCA, BRI, Bank Mandiri, BNI, dan BSI.            
if __name__ == "__main__":
    
    with st.sidebar:
        st.write("<h1 style='text-align: left'><b>DASHBOARD PREDIKSI SAHAM & CRYPTO DENGAN CNN-GRU</b></h1>", unsafe_allow_html=True)
        
        st.write("\n")
        
        st.markdown('**PILIH MENU**')
        
        # Membuat dua opsi menu terpisah
        menu_type = option_menu(
            menu_title=None,
            options=["Informasi Umum", "Prediksi Saham"],
            icons=["info-circle", "graph-up"],
            default_index=0,
            orientation="horizontal"
        )
        
        if menu_type == "Informasi Umum":
            selected = option_menu(
                menu_title=None,
                options=["Gambaran Umum", "Glosarium", "Metodologi"],
                icons=["house", "book", "pen"],
                default_index=0,
                orientation="vertikal"
            )
            
        elif menu_type == "Prediksi Saham":
            selected = option_menu(
                menu_title=None,
                options=["Input Saham Custom", "PT Bank Mandiri Tbk (Bank Mandiri)", "PT Bank Rakyat Indonesia Tbk (BRI)", "PT Bank Central Asia Tbk (BCA)", "PT Bank Negara Indonesia Tbk (BNI)", "PT Bank Syariah Indonesia Tbk (BSI)"],
                icons=["search", "bank", "bank", "bank", "bank", "bank"],
                default_index=0,
                orientation="vertikal"
            )
            
        # Menampilkan Manual
        st.markdown('**Manual**')
        st.markdown('- **1. Pilih Tab Prediksi Saham:** Untuk Melakukan Forecasting')
        st.markdown('- **2. Pilih Option Bank yang tersedia:** Untuk Memilih Kode Saham')
        st.markdown('- **3. Scroll ke bawah halaman:** Untuk Memilih Periode Forecasting')
        st.markdown('- **4. Tekan Tombol Latih Model:** Untuk Melakukan Pelatihan Model Forecasting')
        st.markdown('- **5. Lihat Interpretasi dan Pelaporan Hasil:** Menampilkan Kesimpulan Prediksi Saham')
        
        
        st.write("\n")
        
        with st.expander("Lainnya"):
            st.markdown('**Manual Tambahan**')
            st.markdown('- **Pilih Tab Informasi Umum:** Untuk Mengetahui Informasi Mengenai Aplikasi ini')
            st.markdown('- **Buka Toggle List Persiapan Lingkungan:** Untuk Melihat Library yang digunakan pada Aplikasi ini')
            st.markdown('- **Buka Toggle List Pengumpulan Data:** Untuk Memilih Jumlah Pelatihan Data')
            st.markdown('- **Buka Toggle List Pra-pemrsesan Data:** Untuk Memilih Jumlah Data Pelatihan dan Data Pengujian')
            st.markdown('- **Buka Toggle List Perancangan CNN-GRU:** Untuk Melihat Model Mesin Prediksi yang digunakan')
            st.markdown('- **Buka Toggle List Pelatihan Model:** Untuk Menguubah Hiperparameter, Memilih Beberapa Periode Forecasting dan Melukan Pelatihan Model')
            st.markdown('- **Buka Toggle List Evaluasi Model:** Untuk Melihat Kemampuan Model')
            st.markdown('- **Buka Toggle List Visualisasi Prediksi dan Perhitungan Metrik:** Untuk Melihat Hasil Prediksi Berupa Grafik Dan Perhitungan Metrik')
            st.markdown('- **Buka Toggle List Interpretasi dan Pelaporan Hasil:** Untuk Melihat Interpretasi dan Pelaporan Akhir Periode Forecast yang terakhir')
            

# Logika untuk menampilkan konten berdasarkan pilihan
if menu_type == "Informasi Umum":
    
    if selected == "Gambaran Umum":
        with open('./TEXT/gambaran_umum.md', 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        # Display the HTML content using st.iframe
        st.markdown(html_content, , unsafe_allow_html=True)
        
    elif selected == "Glosarium":
        with open('./TEXT/glosarium.md', 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        # Display the HTML content using st.iframe
        st.markdown(html_content, , unsafe_allow_html=True)
        
    elif selected == "Metodologi":
        with open('./TEXT/metodologi.md', 'r', encoding='utf-8') as file:
            html_content = file.read()
    
        # Display the HTML content using st.iframe
        st.markdown(html_content, , unsafe_allow_html=True)  
        
if menu_type == "Prediksi Saham":
    
    if selected == "Input Saham Custom":
        st.markdown("<h1 style='text-align: left; color: #4A4A4A;'>Input Saham Custom</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify; color: black;'>Masukkan kode saham atau crypto yang ingin Anda prediksi (contoh: AAPL, GOOGL, BMRI.JK, dll.)</p>", unsafe_allow_html=True)
        
        custom_stock = st.text_input("Masukkan Kode Saham", placeholder="Contoh: BMRI.JK")
        
        if custom_stock:
            st.cache_data.clear()
            main(custom_stock)
        else:
            st.warning("Silakan masukkan kode saham terlebih dahulu")
        
    
    if selected == "PT Bank Mandiri Tbk (Bank Mandiri)":
        # Menampilkan logo Perusahaan
        image = Image.open('./LOGO/BMRI.png')
        st.image(image, caption=None, width=500, clamp=False, channels="RGB", output_format="auto")
        
        # Menampilkan Judul
        st.markdown("<h1 style='text-align: left; color: #003A70;'>PT Bank Mandiri Tbk</h1>", unsafe_allow_html=True)
        
        # Menampilkan deskripsi singkat tentang Perusahaan
        st.markdown("<p style='text-align: justify; color: black;'>PT Bank Mandiri (Persero) Tbk adalah salah satu bank BUMN terbesar di Indonesia yang didirikan pada 2 Oktober 1998 sebagai hasil merger 4 bank pemerintah. Bank Mandiri terdaftar di Bursa Efek Indonesia dengan kode saham BMRI. Pemegang saham utamanya adalah Pemerintah Indonesia. Bank ini berfokus pada layanan korporasi, komersial, mikro & ritel, dan tresuri. Bank Mandiri memiliki jaringan cabang dan ATM yang luas di Indonesia serta terus mengembangkan ekosistem digital melalui Livin' by Mandiri.</p>", unsafe_allow_html=True)
        st.write('Informasi singkat:')
        st.markdown('- **Tanggal Didirikan:** 2 Oktober 1998')
        st.markdown('- **Kode Saham Bursa Efek Indonesia:** BMRI')
        st.markdown('- **Pemegang Saham Utama:** Pemerintah Indonesia (66,56%)')
        st.markdown('- **Fokus Pada:** Layanan korporasi, komersial, mikro & ritel, dan tresuri')
        st.markdown('- **Mengembangkan:** Ekosistem digital melalui Livin by Mandiri')
        
        st.cache_data.clear()
        main("BMRI.JK")
 

    elif selected == "PT Bank Rakyat Indonesia Tbk (BRI)":
        # Menampilkan logo Perusahaan
        image = Image.open('./LOGO/BBRI.png')
        st.image(image, caption=None, width=385, clamp=False, channels="RGB", output_format="auto")
        
        # Menampilkan Judul
        st.markdown("<h1 style='text-align: left; color: #00529C;'>PT Bank Rakyat Indonesia Tbk</h1>", unsafe_allow_html=True) 
        
        # Menampilkan deskripsi singkat tentang Perusahaan
        st.markdown("<p style='text-align: justify; color: black;'>PT Bank Rakyat Indonesia (Persero) Tbk (BRI) adalah bank BUMN terbesar di Indonesia yang didirikan pada 16 Desember 1895. BRI terdaftar di Bursa Efek Indonesia dengan kode saham BBRI. Pemegang saham utamanya adalah Pemerintah Indonesia. BRI berfokus utama pada pembiayaan UMKM dan sektor pertanian. Bank ini memiliki jaringan unit kerja terluas hingga ke pelosok desa dan terus mengembangkan layanan perbankan digital seperti BRImo.</p>", unsafe_allow_html=True)
        st.write('Informasi singkat:')
        st.markdown('- **Tanggal Didirikan:** 16 Desember 1895')
        st.markdown('- **Kode Saham Bursa Efek Indonesia:** BBRI')
        st.markdown('- **Pemegang Saham Utama:** Pemerintah Indonesia (53,20%)')
        st.markdown('- **Fokus utama Pada:** Pembiayaan UMKM dan sektor pertanian')
        st.markdown('- **Mengembangkan:** Layanan perbankan digital seperti BRImo')
        
        st.cache_data.clear()
        main("BBRI.JK")

    elif selected == "PT Bank Central Asia Tbk (BCA)":
        # Menampilkan logo Perusahaan
        image = Image.open('./LOGO/BBCA.png')
        st.image(image, caption=None, width=465, clamp=False, channels="RGB", output_format="auto")
        
        # Menampilkan Judul
        st.markdown("<h1 style='text-align: left; color: #0060AF;'>PT Bank Central Asia Tbk</h1>", unsafe_allow_html=True)
        
        # Menampilkan deskripsi singkat tentang Perusahaan
        st.markdown("<p style='text-align: justify; color: black;'>PT Bank Central Asia Tbk adalah bank swasta terbesar di Indonesia yang didirikan pada 21 Februari 1957. BCA terdaftar di Bursa Efek Indonesia dengan kode saham BBCA. Pemegang saham utamanya adalah PT Dwimuria Investama Andalan. BCA berfokus pada layanan perbankan ritel, UKM, dan korporasi. Bank ini memiliki jaringan cabang dan ATM yang luas di seluruh Indonesia serta dikenal dengan layanan perbankan digitalnya seperti m-BCA dan KlikBCA.</p>", unsafe_allow_html=True)
        st.write('Informasi singkat:')
        st.markdown('- **Tanggal Didirikan:** 21 Februari 1957')
        st.markdown('- **Kode Saham Bursa Efek Indonesia:** BBCA')
        st.markdown('- **Pemegang Saham Utama:** PT Dwimuria Investama Andalan (54,94%)')
        st.markdown('- **Fokus Pada:** Layanan perbankan ritel, UKM, dan korporasi')
        st.markdown('- **Dikenal dengan:** Layanan perbankan digital seperti m-BCA dan KlikBCA')
        
        st.cache_data.clear()
        main("BBCA.JK")  

    elif selected == "PT Bank Negara Indonesia Tbk (BNI)":
        # Menampilkan logo Perusahaan
        image = Image.open('./LOGO/BBNI.png')
        st.image(image, caption=None, width=500, clamp=False, channels="RGB", output_format="auto")
        
        # Menampilkan Judul
        st.markdown("<h1 style='text-align: left; color: #006885;'>PT Bank Negara Indonesia Tbk</h1>", unsafe_allow_html=True)
        
        # Menampilkan deskripsi singkat tentang Perusahaan
        st.markdown("<p style='text-align: justify; color: black;'>PT Bank Negara Indonesia (Persero) Tbk adalah salah satu bank BUMN terbesar di Indonesia yang didirikan pada 5 Juli 1946. BNI terdaftar di Bursa Efek Indonesia dengan kode saham BBNI. Pemegang saham utamanya adalah Pemerintah Indonesia. BNI berfokus pada layanan korporasi, ritel, dan internasional. Bank ini memiliki jaringan cabang di dalam dan luar negeri serta terus mengembangkan layanan digital seperti BNI Mobile Banking.</p>", unsafe_allow_html=True)
        st.write('Informasi singkat:')
        st.markdown('- **Tanggal Didirikan:** 5 Juli 1946')
        st.markdown('- **Kode Saham Bursa Efek Indonesia:** BBNI')
        st.markdown('- **Pemegang Saham Utama:** Pemerintah Indonesia (60%)')
        st.markdown('- **Fokus Pada:** Layanan korporasi, ritel, dan internasional')
        st.markdown('- **Mengembangkan:** Layanan digital seperti BNI Mobile Banking')
        
        st.cache_data.clear()
        main("BBNI.JK")

    elif selected == "PT Bank Syariah Indonesia Tbk (BSI)":
        # Menampilkan logo Perusahaan
        image = Image.open('./LOGO/BRIS.png')
        st.image(image, caption=None, width=520, clamp=False, channels="RGB", output_format="auto")
        
        # Menampilkan Judul
        st.markdown("<h1 style='text-align: left; color: #00A39D;'>PT Bank Syariah Indonesia Tbk</h1>", unsafe_allow_html=True)
        
        # Menampilkan deskripsi singkat tentang Perusahaan
        st.markdown("<p style='text-align: justify; color: black;'>PT Bank Syariah Indonesia Tbk adalah bank syariah terbesar di Indonesia yang didirikan pada 1 Februari 2021 sebagai hasil merger 3 bank syariah BUMN. BSI terdaftar di Bursa Efek Indonesia dengan kode saham BRIS. Pemegang saham utamanya adalah PT Bank Mandiri (Persero) Tbk, PT Bank Negara Indonesia (Persero) Tbk, dan PT Bank Rakyat Indonesia (Persero) Tbk. BSI berfokus pada layanan perbankan syariah ritel dan korporasi serta terus mengembangkan ekosistem keuangan syariah digital.</p>", unsafe_allow_html=True)
        st.write('Informasi singkat:')
        st.markdown('- **Tanggal Didirikan:** 1 Februari 2021 (hasil merger 3 bank syariah BUMN)')
        st.markdown('- **Kode Saham Bursa Efek Indonesia:** BRIS')
        st.markdown('- **Pemegang Saham Utama:** PT Bank Mandiri (Persero) Tbk (51,47%), PT Bank Negara Indonesia (Persero) Tbk (23,24%), PT Bank Rakyat Indonesia (Persero) Tbk (15,38%)')
        st.markdown('- **Fokus Pada:** Layanan perbankan syariah ritel dan korporasi')
        st.markdown('- **Mengembangkan:** Ekosistem keuangan syariah digital')
        
        st.cache_data.clear()
        main("BRIS.JK")
