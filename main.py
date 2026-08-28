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

# Helper function to safely extract scalar from pandas Series
def safe_float(value):
    if hasattr(value, 'item'):
        return float(value.item())
    return float(value)

def smart_format(val, default_decimals=2, prefix=""):
    """
    Format angka secara cerdas dengan dukungan sub-angka (subscript) untuk 0.000...
    - Jika val adalah None / NaN: kembalikan "-"
    - Jika val == 0: kembalikan prefix + "0.00"
    - Jika abs(val) >= 0.01: format standar (misal 1,234.56 atau 0.05)
    - Jika 0 < abs(val) < 0.01: format 0.0_{subscript}N... (misal 0.0₄1234 untuk 0.00001234)
    """
    if val is None or (isinstance(val, (float, np.floating, int, np.integer)) and pd.isna(val)):
        return "-"
    try:
        fval = float(val)
    except (ValueError, TypeError):
        return str(val)

    if fval == 0:
        return f"{prefix}0.00"

    sign = "-" if fval < 0 else ""
    abs_val = abs(fval)

    if abs_val >= 0.01:
        if abs_val >= 1000:
            return f"{prefix}{sign}{abs_val:,.2f}"
        elif default_decimals == 3:
            return f"{prefix}{sign}{abs_val:.3f}"
        elif default_decimals == 4:
            return f"{prefix}{sign}{abs_val:.4f}"
        else:
            return f"{prefix}{sign}{abs_val:.2f}"

    # Untuk nilai sangat kecil: 0 < abs_val < 0.01
    exp_str = f"{abs_val:.10e}"
    parts = exp_str.split('e')
    mantissa = float(parts[0])
    exponent = int(parts[1])

    zero_count = abs(exponent) - 1

    subscript_digits = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'}
    sub_str = ''.join(subscript_digits[d] for d in str(zero_count))

    sig_digits = f"{mantissa:.4f}".replace('.', '').rstrip('0')
    if not sig_digits:
        sig_digits = f"{mantissa:.2f}".replace('.', '')

    return f"{prefix}{sign}0.0{sub_str}{sig_digits}"

def format_df_for_display(df):
    if df is None or df.empty:
        return df
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            display_df[col] = display_df[col].apply(lambda x: smart_format(x))
    return display_df

@st.cache_data(ttl=3600)
def get_market_cap(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        cap = info.get('marketCap', None)
        if cap is None or cap == 0:
            cap = info.get('regularMarketVolume', None)
        return cap
    except Exception:
        return None

def format_market_cap(cap, curr_prefix=""):
    if cap is None or cap == 0 or pd.isna(cap):
        return "-"
    try:
        fcap = float(cap)
        if fcap >= 1e12:
            return f"{curr_prefix}{fcap/1e12:,.2f} T"
        elif fcap >= 1e9:
            return f"{curr_prefix}{fcap/1e9:,.2f} B"
        elif fcap >= 1e6:
            return f"{curr_prefix}{fcap/1e6:,.2f} M"
        else:
            return smart_format(fcap, prefix=curr_prefix)
    except Exception:
        return "-"

def main(stock):
    if 'current_stock' not in st.session_state:
        st.session_state.current_stock = ""
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False

    if st.session_state.current_stock != stock:
        if st.session_state.training_completed:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.training_completed = False
        st.session_state.current_stock = stock

    st.header(f"Prediksi Harga Saham dengan kode {stock}")

    # Ringkasan 3 kolom di bawah header
    curr_prefix = "$ " if (stock.endswith('-USD') or stock.endswith('-IDR') or 'USD' in stock) else "Rp "
    try:
        if crypto_yfinance and (stock.endswith('-USD') or stock.endswith('-IDR') or stock in ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'DOT-USD', 'SHIB-USD', 'AVAX-USD']):
            quick_df = cyf.download(stock, start="2020-01-01", end=date.today().strftime("%Y-%m-%d"))
        else:
            quick_df = yf.download(stock, start="2020-01-01", end=date.today().strftime("%Y-%m-%d"))
        
        if not quick_df.empty:
            if 'Date' in quick_df.columns:
                last_dt = pd.to_datetime(quick_df['Date'].iloc[-1]).strftime('%Y-%m-%d')
            else:
                last_dt = pd.to_datetime(quick_df.index[-1]).strftime('%Y-%m-%d')
            last_pr = safe_float(quick_df['Close'].iloc[-1])
            last_pr_str = smart_format(last_pr, prefix=curr_prefix)
        else:
            last_dt = "-"
            last_pr_str = "-"
    except Exception:
        last_dt = "-"
        last_pr_str = "-"

    mcap_val = get_market_cap(stock)
    mcap_str = format_market_cap(mcap_val, curr_prefix=curr_prefix)

    col_sum1, col_sum2, col_sum3 = st.columns(3)
    with col_sum1:
        st.metric("Tanggal Terakhir", last_dt)
    with col_sum2:
        st.metric("Harga Terakhir", last_pr_str)
    with col_sum3:
        st.metric("Market Cap", mcap_str)
    st.write("")

    with st.expander("1. Persiapan Lingkungan"):
        with st.spinner("Mengimpor library yang diperlukan..."):
            with st.popover("Detail Library yang Digunakan"):
                st.info('Import Library Utama: `time`, `numpy`, `pandas`, `yfinance`, `PIL` untuk pengelolaan data dasar, manipulasi array, serta penampilan gambar.', icon=":material/code:")
                st.warning('Import Framework ML & Visualisasi: `streamlit`, `tensorflow`, `matplotlib`, `sklearn`, `streamlit_option_menu` untuk pemodelan CNN-GRU dan antarmuka web interaktif.', icon=":material/extension:")
            
            # Workflow Diagram
            st.subheader("Diagram Alur Kerja")
            mermaid_code = """
            graph TD
                A[Input Saham/Crypto] --> B[Pengumpulan Data]
                B --> C[Pra-pemrosesan Data]
                C --> D[Perancangan Model CNN-GRU]
                D --> E[Pelatihan Model]
                E --> F[Evaluasi Model]
                F --> G[Visualisasi Prediksi]
                G --> H[Interpretasi Hasil]
            """
            st.code(mermaid_code, language='mermaid')
            st.success("Library berhasil diimpor")

    with st.expander("2. Pengumpulan Data"):

        #Menyimpan Data pada Cache
        @st.cache_data
        def load_data(ticker, start_date, end_date):
            try:
                if crypto_yfinance and (ticker.endswith('-USD') or ticker.endswith('-IDR') or ticker in ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'DOT-USD', 'SHIB-USD', 'AVAX-USD']):
                    data = cyf.download(ticker, start=start_date, end=end_date)
                else:
                    data = yf.download(ticker, start=start_date, end=end_date)
                
                if data.empty:
                    st.error(f"Tidak dapat memuat data untuk {ticker}. Silakan coba simbol lain.")
                    return pd.DataFrame()
                    
                data.reset_index(inplace=True)
                
                if 'Date' not in data.columns:
                    if 'Datetime' in data.columns:
                        data.rename(columns={'Datetime': 'Date'}, inplace=True)
                    else:
                        data['Date'] = data.index
                        data.reset_index(drop=True, inplace=True)
                    
                if 'Date' in data.columns:
                    data['Date'] = pd.to_datetime(data['Date'])
                    
                return data
            except Exception as e:
                st.error(f"Error loading data for {ticker}: {str(e)}")
                return pd.DataFrame()

        # DATA HISTORY
        full_data = load_data(stock, "2000-01-01", date.today().strftime("%Y-%m-%d"))

        st.subheader("Data keseluruhan")
        st.write("Mulai")
        st.write(format_df_for_display(full_data.head(1)))
        st.write("Hingga")
        st.write(format_df_for_display(full_data.tail(1)))

        # Mengubah index menjadi datetime untuk memudahkan plotting
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
            st.write(format_df_for_display(full_data))

        # DATA PELATIHAN
        # Pilihan untuk input jumlah data pelatihan
        st.subheader("Pengaturan Data Pelatihan")

        use_today_end = st.checkbox("Gunakan hingga tanggal terbaru (Hari ini)", value=True)
        if use_today_end:
            end_date_obj = date.today()
        else:
            end_date_obj = st.date_input(
                "Tanggal Selesai Pelatihan",
                value=date.today(),
                min_value=date(2000, 1, 1),
                max_value=date.today()
            )

        method_options = ["Rentang Tahun / Bulan / Hari", "Gunakan Jumlah Hari Terakhir", "Pilih Tanggal dengan Kalender"]
        selected_method = st.radio(
            "Pilihan Metode Memilih Data Pelatihan:",
            options=method_options,
            index=0
        )

        if selected_method == "Rentang Tahun / Bulan / Hari":
            years_ago = st.slider('Pilih berapa tahun yang lalu untuk pelatihan:', 0, 30, 4)
            months_ago = st.slider('Pilih berapa bulan tambahan yang lalu untuk pelatihan:', 0, 11, 0 if stock in ["BBCA.JK", "BMRI.JK", "BBNI.JK"] else 1)
            days_ago = st.slider('Pilih berapa hari tambahan yang lalu untuk pelatihan:', 0, 31, 0)
            
            days = (years_ago * 365) + (months_ago * 30) + days_ago
            if days < 120:
                days = 120
            start_date_obj = end_date_obj - timedelta(days=days)

        elif selected_method == "Gunakan Jumlah Hari Terakhir":
            default_val = 1800 if stock in ["BBCA.JK", "BMRI.JK", "BBNI.JK"] else 1470
            days = st.number_input("Jumlah hari untuk pelatihan", min_value=120, max_value=365*30, value=default_val)
            start_date_obj = end_date_obj - timedelta(days=days)

        else: # "Pilih Tanggal dengan Kalender"
            default_start = end_date_obj - timedelta(days=1470) # 1470 hari yang lalu (~4 tahun 1 bulan yang lalu)
            
            start_date_selected = st.date_input(
                "Tanggal Mulai Pelatihan",
                value=default_start,
                min_value=date(2000, 1, 1),
                max_value=end_date_obj
            )
            
            start_date_obj = start_date_selected
            days = (end_date_obj - start_date_obj).days
            if days < 120:
                days = 120

        with st.popover("Tips Memilih Data Pelatihan"):
            st.info('Ket: Semakin lama hari yang dipilih, maka jumlah hari Prediksi dapat dilakukan dengan lebih banyak, namun prediksi menjadi lebih tidak akurat.', icon=":material/notes:")
            st.warning('Ket: Secara Default menggunakan 4 Tahun 1 Bulan atau 1470 hari yang lalu, yang hanya dapat melakukan prediksi hingga 6 Bulan kedepan.', icon=":material/pan_tool_alt:")
            st.warning('Ket: Jika menggunakan Kalender, secara default tanggal mulai diset ke 1470 hari yang lalu (sekitar 4 Tahun 1 Bulan) dan tanggal selesai diset ke tanggal terbaru.', icon=":material/calendar_month:")
            st.warning('Ket: Jumlah Minimal 4 Bulan atau 120 hari yang lalu, yang hanya dapat melakukan prediksi hingga 3 hari kedepan, Perhatikanlah pada bagian Pra-pemrosesan data "Ukuran data pengujian" jumlahnya sebanding dengan jumlah hari yang dapat anda lakukan untuk prediksi kedepan.', icon=":material/exclamation:")

        # Mengubah Format Tanggal data Pelatihan
        start_date = start_date_obj.strftime("%Y-%m-%d")
        end_date = end_date_obj.strftime("%Y-%m-%d")

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

        # Mengubah index menjadi datetime untuk data pelatihan
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        if not data.empty:
            actual_days = (data.index[-1] - data.index[0]).days
            y_cnt = actual_days // 365
            rem_days = actual_days % 365
            m_cnt = rem_days // 30
            d_cnt = rem_days % 30
            duration_str = f"{y_cnt} Tahun {m_cnt} Bulan {d_cnt} Hari"
        else:
            actual_days = 0
            duration_str = "0 Tahun 0 Bulan 0 Hari"

        st.subheader("Data Pelatihan yang telah dipilih")
        st.write(f"Jumlah Hari yang dipilih **{actual_days}** ({duration_str}).")
        st.write("Mulai")
        st.write(format_df_for_display(data.head(1)))
        st.write("Hingga")
        st.write(format_df_for_display(data.tail(1)))

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
            st.write(format_df_for_display(data))

    with st.expander("3. Pra-pemrosesan Data"):

        if days >= 120:

            with st.popover("⚙️ Pengaturan Panjang Sekuens (Lookback)"):
                seq_options = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180]
                max_seq = max(5, min(180, len(data) // 3))
                valid_seq_options = [s for s in seq_options if s <= max_seq]
                if not valid_seq_options:
                    valid_seq_options = [30]
                default_seq = 60 if 60 in valid_seq_options else valid_seq_options[-1]
                seq_length = st.select_slider("Panjang Sekuens (Hari)", options=valid_seq_options, value=default_seq)
                st.info('Rekomendasi Default: **60 Hari** (~3 Bulan bursa). Rentang Paling Akurat: **30–60 Hari**.', icon=":material/recommend:")
                st.warning('Ket: Sekuens terlalu pendek (<15) kehilangan konteks tren, sekuens terlalu panjang (>120) menambah dimensi & mengurangi jumlah sampel data.', icon=":material/timeline:")

            with st.popover("⚙️ Pengaturan Normalisasi (MinMaxScaler)"):
                scaler_option = st.radio("Rentang Normalisasi:", options=["(0, 1)", "(-1, 1)"], index=0)
                feature_range = (0, 1) if scaler_option == "(0, 1)" else (-1, 1)
                st.info('Rekomendasi Default: **(0, 1)**. Sangat optimal untuk aktivasi ReLU dan harga aset non-negatif.', icon=":material/recommend:")
                st.warning('Ket: Rentang (-1, 1) dapat digunakan jika menggunakan aktivasi simetris tanh di seluruh model.', icon=":material/tune:")

            with st.popover("⚙️ Pengaturan Pembagian Data (Train/Test Split)"):
                split_pct = st.select_slider("Persentase Data Pelatihan (%)", options=[50, 60, 70, 75, 80, 85, 90], value=80)
                train_ratio = split_pct / 100.0
                st.info('Rekomendasi Default: **80% Pelatihan : 20% Pengujian** (Rentang Paling Akurat: **75%–85%**).', icon=":material/recommend:")
                st.warning('Ket: Porsi pelatihan <70% membuat model kurang belajar, porsi >85% membuat data pengujian terlalu sedikit untuk validasi.', icon=":material/pie_chart:")

            def preprocess_data(data, seq_length, feature_range, train_ratio):
                scaler = MinMaxScaler(feature_range=feature_range)
                scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

                X, y = [], []
                for i in range(seq_length, len(scaled_data)):
                    X.append(scaled_data[i-seq_length:i, 0])
                    y.append(scaled_data[i, 0])

                X, y = np.array(X), np.array(y)

                split = int(train_ratio * len(X))
                x_train, x_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
                x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

                return x_train, x_test, y_train, y_test, scaler

            x_train, x_test, y_train, y_test, scaler = preprocess_data(data, seq_length, feature_range, train_ratio)

            # Menghitung persentase data pelatihan dan pengujian
            total_samples = x_train.shape[0] + x_test.shape[0]
            train_percentage = (x_train.shape[0] / total_samples) * 100
            test_percentage = (x_test.shape[0] / total_samples) * 100

            with st.popover("Detail Pra-pemrosesan Data"):
                st.info(f'Ukuran Panjang Sekuens (`seq_length`): Menggunakan **{seq_length}** hari historis untuk memprediksi harga saham pada hari berikutnya.', icon=":material/timeline:")
                st.warning(f'Pembagian Data: Data di-scaling dengan `MinMaxScaler{feature_range}` lalu dibagi menjadi **{train_percentage:.1f}%** data pelatihan ({x_train.shape[0]} sampel) dan **{test_percentage:.1f}%** data pengujian ({x_test.shape[0]} sampel).', icon=":material/pie_chart:")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ukuran data pelatihan", f"{x_train.shape[0]} sampel")
                st.metric("Persentase data pelatihan", f"{train_percentage:.2f}%")
            with col2:
                st.metric("Ukuran data pengujian", f"{x_test.shape[0]} sampel")
                st.metric("Persentase data pengujian", f"{test_percentage:.2f}%")

            st.success("Pra-pemrosesan Data selesai!")
        else:
            st.warning('Harus Memilih Jumlah Hari Minimal 4 Bulan atau 120 hari', icon=":material/exclamation:")

    with st.expander("4. Perancangan Model CNN-GRU"):

        if days >= 120:

            st.subheader("Arsitektur Model & Pengaturan Hyperparameter:")

            with st.popover("⚙️ Pengaturan Lapisan Conv1D"):
                conv_filters = st.select_slider("Jumlah Filter Conv1D", options=[16, 32, 48, 64, 96, 128, 256], value=64)
                kernel_size = st.select_slider("Ukuran Kernel (Kernel Size)", options=[2, 3, 4, 5, 7], value=3)
                conv_activation = st.selectbox("Fungsi Aktivasi Conv1D", options=["relu", "tanh", "elu", "linear"], index=0)
                st.info('Rekomendasi Default: **64 filter**, **kernel 3**, aktivasi **ReLU** (Rentang Akurat: 32–64 filter, kernel 3–5).', icon=":material/recommend:")
                st.warning('Ket: Filter Conv1D mengekstrak fitur spasial & momentum lokal jangka pendek dari sekuens harga.', icon=":material/layers:")

            with st.popover("⚙️ Pengaturan Lapisan GRU"):
                gru_units_1 = st.select_slider("Unit GRU Layer 1", options=[16, 32, 50, 64, 96, 128, 256], value=50)
                gru_units_2 = st.select_slider("Unit GRU Layer 2", options=[16, 32, 50, 64, 96, 128, 256], value=50)
                st.info('Rekomendasi Default: **50 Unit** per layer (Rentang Akurat: **32–64 Unit**).', icon=":material/recommend:")
                st.warning('Ket: GRU menangkap ketergantungan temporal jangka panjang. Nilai di atas 128 meningkatkan risiko overfitting pada deret waktu.', icon=":material/memory:")

            with st.popover("⚙️ Pengaturan Regularisasi Dropout"):
                dropout_rate = st.select_slider("Tingkat Dropout (Dropout Rate)", options=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5], value=0.2)
                st.info('Rekomendasi Default: **0.2 (20%)** (Rentang Akurat: **0.1–0.3**).', icon=":material/recommend:")
                st.warning('Ket: Dropout 0.2 mencegah ko-adaptasi neuron yang menyebabkan overfitting tanpa membuang representasi penting.', icon=":material/security:")

            def create_model(seq_len, c_filters, k_size, c_act, g_u1, g_u2, d_rate):
                model = Sequential([
                    Conv1D(filters=c_filters,
                        kernel_size=k_size,
                        activation=c_act,
                        input_shape=(seq_len, 1)),
                    GRU(g_u1, return_sequences=True),
                    Dropout(d_rate),
                    GRU(g_u2),
                    Dense(1)
                ])

                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                return model

            # Buat model untuk inspeksi parameter sebelum pelatihan
            preview_model = create_model(seq_length, conv_filters, kernel_size, conv_activation, gru_units_1, gru_units_2, dropout_rate)
            total_params = preview_model.count_params()

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Total Parameter Model (Trainable)", f"{total_params:,}")
            with col_p2:
                st.metric("Jumlah Lapisan (Layers)", f"{len(preview_model.layers)} Lapisan")

            with st.popover("Detail Arsitektur & Ringkasan Parameter"):
                st.info(f'Lapisan Ekstraksi Fitur: `Conv1D` ({conv_filters} filter, kernel {kernel_size}, aktivasi {conv_activation}) untuk mengekstrak pola fitur spasial dari sekuens {seq_length} hari.', icon=":material/layers:")
                st.warning(f'Lapisan Memori & Regulasi: `GRU` ({gru_units_1} unit, seq) $\\rightarrow$ `Dropout` ({dropout_rate}) $\\rightarrow$ `GRU` ({gru_units_2} unit) untuk menangkap pola temporal sekuensial.', icon=":material/memory:")
                st.warning('Lapisan Output & Kompilasi: Lapisan `Dense(1)` dikompilasi dengan optimizer `Adam(learning_rate=0.001)` dan loss function `MSE`.', icon=":material/check_circle:")
                
                layer_data = []
                for lyr in preview_model.layers:
                    layer_data.append({
                        "Nama Lapisan": lyr.name,
                        "Tipe Lapisan": lyr.__class__.__name__,
                        "Bentuk Output": str(lyr.output_shape),
                        "Jumlah Parameter": f"{lyr.count_params():,}"
                    })
                st.dataframe(pd.DataFrame(layer_data), use_container_width=True)

            def get_model(seq_len):
                return create_model(seq_len, conv_filters, kernel_size, conv_activation, gru_units_1, gru_units_2, dropout_rate)

            st.success("Perancangan Model CNN-GRU selesai!")

        else:
            st.warning('Harus Memilih Jumlah Hari Minimal 4 Bulan atau 120 hari', icon=":material/exclamation:")

    with st.expander("5. Pelatihan Model", True):

        # Menambahkan nilai default untuk ketika tombol belum ditekan
        btn_check = 0

        if days >= 120:

            with st.popover("Mengubah Jumlah Epoch"):
                epoch_options = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300, 500]
                epochs = st.select_slider("Jumlah Epoch", options=epoch_options, value=50)
                st.info('Rekomendasi Default: **50 Epoch** (Rentang Paling Akurat: **40–100 Epoch**).', icon=":material/recommend:")
                st.warning('Ket: Semakin banyak Jumlahnya (>150), maka waktu komputasi semakin lambat dan berisiko overfitting pada noise pasar.', icon=":material/timer_3_alt_1:")
                st.info('Ket: Semakin sedikit Jumlahnya (<20), maka komputasi sangat cepat namun berisiko underfitting (bobot GRU belum konvergen).', icon=":material/speed:")

            with st.popover("Mengubah Ukuran Batch"):
                batch_size_options = [2, 4, 8, 16, 32, 64, 128, 256, 512]
                batch_size = st.select_slider("Ukuran Batch", options=batch_size_options, value=32)
                st.info('Rekomendasi Default: **32** (Rentang Paling Akurat: **16–32** untuk Time Series CNN-GRU).', icon=":material/recommend:")
                st.warning('Ket: Ukuran kecil (4–16) memberikan regularisasi stokastik lebih baik namun komputasi relatif lebih lambat.', icon=":material/tune:")
                st.info('Ket: Ukuran besar (128–512) mempercepat komputasi GPU/CPU namun rentan mengalami generalization gap pada data deret waktu.', icon=":material/bolt:")

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
                    st.session_state.training_completed = True

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
                    st.metric("MSE", smart_format(mse, default_decimals=3))
                    st.metric("RMSE", smart_format(rmse, default_decimals=3))
                    st.metric("R2 Score", smart_format(r2, default_decimals=3))
                    st.metric("MAPE", smart_format(mape, default_decimals=3))
                    st.metric("Akurasi", f"{smart_format(100 - mape*100, default_decimals=3)}%")

                with col2:
                    # Menampilkan tabel perbandingan
                    comparison_df = pd.DataFrame({
                        'Tanggal': actual_dates.strftime('%Y-%m-%d'),
                        'Harga Aktual': [smart_format(v) for v in y_test],
                        'Harga Prediksi': [smart_format(v) for v in y_pred]
                    })
                    st.dataframe(comparison_df)

                accuracy = 100 - mape * 100
                if accuracy < 0:
                    if accuracy >= -50:
                        st.info('Performa: Kurang Baik (Akurasi Negatif)', icon=":material/thumb_down:")
                    elif accuracy >= -100:
                        st.error('Performa: Buruk (Akurasi Negatif)', icon=":material/thumb_down:")
                    else:
                        st.error('Performa: Sangat Buruk (Akurasi Negatif)', icon=":material/thumb_down:")
                elif isinstance(rmse, (int, float)):
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
                        st.metric("Loss akhir", smart_format(final_loss, default_decimals=4))
                        st.metric("Validation Loss akhir", smart_format(final_val_loss, default_decimals=4))

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
                    last_actual_price = safe_float(data['Close'].iloc[-1])  # Extract scalar value safely
                    last_forecast_price = safe_float(forecast[-1][0])  # Extract scalar value safely
                    percent_change = ((last_forecast_price - last_actual_price) / last_actual_price) * 100

                    if len(y_test) >= forecast_days:

                        st.subheader("Tabel dan Metrik Performa:")

                        table_df = pd.DataFrame({
                                    'Tanggal': date_range.strftime('%Y-%b-%d'),
                                    'Harga Prediksi': [smart_format(v) for v in forecast.flatten()]
                                })

                        # Calculate metrics
                        mse = mean_squared_error(y_test[:forecast_days], y_pred[:forecast_days])
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test[:forecast_days], y_pred[:forecast_days])
                        mape = mean_absolute_percentage_error(y_test[:forecast_days], y_pred[:forecast_days])
                        accuracy = 100 - mape * 100

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("MSE", smart_format(mse, default_decimals=3))
                            st.metric("MAPE", smart_format(mape, default_decimals=3))
                            # Menampilkan tabel perbandingan
                            with st.popover("Tampilkan Tabel"):
                                st.dataframe(table_df)
                        with col2:
                            st.metric("RMSE", smart_format(rmse, default_decimals=3))
                            st.metric("R2 Score", smart_format(r2, default_decimals=3))
                            st.metric("Akurasi", f"{smart_format(accuracy, default_decimals=3)}%")

                        st.subheader("Ringkasan Prediksi")

                        curr_prefix = "$ " if (stock.endswith('-USD') or stock.endswith('-IDR') or 'USD' in stock) else "Rp "
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Harga Terakhir", smart_format(last_actual_price, prefix=curr_prefix))
                        with col2:
                            st.metric("Prediksi Harga", smart_format(last_forecast_price, prefix=curr_prefix), f"{percent_change:.2f}%")

                        if accuracy < 0:
                            if accuracy >= -50:
                                st.info('Performa: Kurang Baik (Akurasi Negatif)', icon=":material/thumb_down:")
                            elif accuracy >= -100:
                                st.error('Performa: Buruk (Akurasi Negatif)', icon=":material/thumb_down:")
                            else:
                                st.error('Performa: Sangat Buruk (Akurasi Negatif)', icon=":material/thumb_down:")
                        elif rmse < 50:
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
            st.warning('Harus Melakukan Pelatihan Model Terlebih dahulu', icon=":material/exclamation:")

    with st.expander("8. Interpretasi dan Pelaporan Hasil", True):

        # Mengecek apakah sudah menekan tombol Latih Model
        if btn_check == 1:

            if 'last_actual_price' in locals() and 'last_forecast_price' in locals() and 'percent_change' in locals():

                st.subheader(f"Ringkasan Prediksi **{forecast_period}** ke depan")
                curr_prefix = "$ " if (stock.endswith('-USD') or stock.endswith('-IDR') or 'USD' in stock) else "Rp "
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Harga Terakhir", smart_format(last_actual_price, prefix=curr_prefix))
                with col2:
                    st.metric("Prediksi Harga", smart_format(last_forecast_price, prefix=curr_prefix), f"{percent_change:.2f}%")

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

                accuracy = 100 - mape * 100
                if accuracy < 0:
                    if accuracy >= -50:
                        st.info('Performa: Kurang Baik (Akurasi Negatif)', icon=":material/thumb_down:")
                    elif accuracy >= -100:
                        st.error('Performa: Buruk (Akurasi Negatif)', icon=":material/thumb_down:")
                    else:
                        st.error('Performa: Sangat Buruk (Akurasi Negatif)', icon=":material/thumb_down:")
                elif rmse < 50:
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
            
            with st.expander("💡 Rekomendasi & Catatan Penting", expanded=False):
                st.markdown("**Rekomendasi:**")
                recommendation = generate_recommendation(percent_change, accuracy)
                st.write(recommendation)
                
                st.markdown("**Catatan Penting:**")
                st.warning("""
- Prediksi ini didasarkan pada data historis dan model statistik.
- Faktor eksternal seperti kondisi ekonomi, kebijakan perusahaan, dan peristiwa global dapat mempengaruhi harga saham secara signifikan.
- Selalu lakukan analisis tambahan dan konsultasikan dengan penasihat keuangan sebelum membuat keputusan investasi.
                """, icon=":material/edit_note:")
        else:
            st.warning('Harus Melakukan Pelatihan Model Terlebih dahulu', icon=":material/exclamation:")

# Pengguna memilih bank yang ingin dianalisis dari sidebar Streamlit. Pilihan bank termasuk BCA, BRI, Bank Mandiri, BNI, dan BSI.            
if __name__ == "__main__":
    
    # Check programmatic redirection
    manual_select_type = None
    if st.session_state.get('redirect_to_input_custom', False):
        st.session_state.menu_type_index = 1
        manual_select_type = 1
        st.session_state.redirect_to_input_custom = False
        
    if 'menu_type_index' not in st.session_state:
        st.session_state.menu_type_index = 0

    with st.sidebar:
        st.write("<h1 style='text-align: left'><b>DASHBOARD PREDIKSI SAHAM & CRYPTO DENGAN CNN-GRU</b></h1>", unsafe_allow_html=True)
        
        st.write("\n")
        
        st.markdown('**PILIH MENU**')
        
        # Membuat dua opsi menu terpisah
        menu_type = option_menu(
            menu_title=None,
            options=["Informasi Umum", "Prediksi Saham"],
            icons=["info-circle", "graph-up"],
            default_index=st.session_state.menu_type_index,
            manual_select=manual_select_type,
            orientation="horizontal"
        )
        
        # Sync menu_type_index to avoid sticking
        menu_type_options = ["Informasi Umum", "Prediksi Saham"]
        if menu_type in menu_type_options:
            st.session_state.menu_type_index = menu_type_options.index(menu_type)
        
        if menu_type == "Informasi Umum":
            if 'selected_index_info' not in st.session_state:
                st.session_state.selected_index_info = 0
                
            selected = option_menu(
                menu_title=None,
                options=["Gambaran Umum", "Glosarium", "Metodologi"],
                icons=["house", "book", "pen"],
                default_index=st.session_state.selected_index_info,
                orientation="vertikal"
            )
            
            # Sync selection back
            info_options = ["Gambaran Umum", "Glosarium", "Metodologi"]
            if selected in info_options:
                st.session_state.selected_index_info = info_options.index(selected)
            
        elif menu_type == "Prediksi Saham":
            manual_select_pred = None
            if st.session_state.get('redirect_to_input_custom_pred', False):
                st.session_state.selected_index_pred = 0
                manual_select_pred = 0
                st.session_state.redirect_to_input_custom_pred = False
                
            if 'selected_index_pred' not in st.session_state:
                st.session_state.selected_index_pred = 0
                
            selected = option_menu(
                menu_title=None,
                options=["Input Saham Custom", "PT Bank Mandiri Tbk (Bank Mandiri)", "PT Bank Rakyat Indonesia Tbk (BRI)", "PT Bank Central Asia Tbk (BCA)", "PT Bank Negara Indonesia Tbk (BNI)", "PT Bank Syariah Indonesia Tbk (BSI)"],
                icons=["search", "bank", "bank", "bank", "bank", "bank"],
                default_index=st.session_state.selected_index_pred,
                manual_select=manual_select_pred,
                orientation="vertikal"
            )
            
            # Sync selection back
            pred_options = ["Input Saham Custom", "PT Bank Mandiri Tbk (Bank Mandiri)", "PT Bank Rakyat Indonesia Tbk (BRI)", "PT Bank Central Asia Tbk (BCA)", "PT Bank Negara Indonesia Tbk (BNI)", "PT Bank Syariah Indonesia Tbk (BSI)"]
            if selected in pred_options:
                st.session_state.selected_index_pred = pred_options.index(selected)
            
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
        
        # Split after the portfolio paragraph and inject redirect button
        target_marker = "Portofolio ini dibuat oleh Ilham Rizkyansyah &middot; Universitas Gunadarma Informatika"
        if target_marker in html_content:
            parts = html_content.split(target_marker)
            subparts = parts[1].split("</p>", 1)
            
            # Display first part up to the paragraph close tag
            st.markdown(parts[0] + target_marker + subparts[0] + "</p>", unsafe_allow_html=True)
            
            # Display shortcut button
            if st.button("👉 Klik di sini untuk melakukan forecasting langsung", type="primary", use_container_width=True):
                st.session_state.redirect_to_input_custom = True
                st.session_state.redirect_to_input_custom_pred = True
                st.rerun()
            
            # Display remaining markdown content
            st.markdown(subparts[1], unsafe_allow_html=True)
        else:
            st.markdown(html_content, unsafe_allow_html=True)
        
    elif selected == "Glosarium":
        with open('./TEXT/glosarium.md', 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        # Display the HTML content using st.iframe
        st.markdown(html_content, unsafe_allow_html=True)
        
    elif selected == "Metodologi":
        with open('./TEXT/metodologi.md', 'r', encoding='utf-8') as file:
            html_content = file.read()
    
        # Display the HTML content using st.iframe
        st.markdown(html_content, unsafe_allow_html=True)  
        
if menu_type == "Prediksi Saham":
    
    if selected == "Input Saham Custom":
        if 'custom_stock_input' not in st.session_state:
            st.session_state.custom_stock_input = ""

        st.markdown("<h1 style='text-align: left; color: #4A4A4A;'>Input Saham Custom</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify; color: black;'>Masukkan kode saham atau crypto yang ingin Anda prediksi (contoh: AAPL, GOOGL, BMRI.JK, dll.)</p>", unsafe_allow_html=True)
        st.info("💡 **Tips:** Silakan scroll ke bawah halaman untuk melakukan forecasting/prediksi setelah memilih atau memasukkan kode saham.")
        
        # Popular Indonesia Stock Buttons
        st.write("**Pilihan Saham Indonesia Populer**")
        ind_cols = st.columns(4)
        with ind_cols[0]:
            if st.button("BBCA.JK", key="btn_bbca"):
                st.session_state.custom_stock_input = "BBCA.JK"
                st.rerun()
        with ind_cols[1]:
            if st.button("BMRI.JK", key="btn_bmri"):
                st.session_state.custom_stock_input = "BMRI.JK"
                st.rerun()
        with ind_cols[2]:
            if st.button("BBNI.JK", key="btn_bbni"):
                st.session_state.custom_stock_input = "BBNI.JK"
                st.rerun()
        with ind_cols[3]:
            if st.button("BRIS.JK", key="btn_bris"):
                st.session_state.custom_stock_input = "BRIS.JK"
                st.rerun()
        
        # Popular Global Stock Buttons
        st.write("**Pilihan Saham Global Populer**")
        glob_cols1 = st.columns(4)
        with glob_cols1[0]:
            if st.button("AAPL", key="btn_aapl"):
                st.session_state.custom_stock_input = "AAPL"
                st.rerun()
        with glob_cols1[1]:
            if st.button("GOOGL", key="btn_googl"):
                st.session_state.custom_stock_input = "GOOGL"
                st.rerun()
        with glob_cols1[2]:
            if st.button("MSFT", key="btn_msft"):
                st.session_state.custom_stock_input = "MSFT"
                st.rerun()
        with glob_cols1[3]:
            if st.button("TSLA", key="btn_tsla"):
                st.session_state.custom_stock_input = "TSLA"
                st.rerun()
        
        glob_cols2 = st.columns(4)
        with glob_cols2[0]:
            if st.button("AMZN", key="btn_amzn"):
                st.session_state.custom_stock_input = "AMZN"
                st.rerun()
        with glob_cols2[1]:
            if st.button("NVDA", key="btn_nvda"):
                st.session_state.custom_stock_input = "NVDA"
                st.rerun()
        with glob_cols2[2]:
            if st.button("META", key="btn_meta"):
                st.session_state.custom_stock_input = "META"
                st.rerun()
        with glob_cols2[3]:
            if st.button("NFLX", key="btn_nflx"):
                st.session_state.custom_stock_input = "NFLX"
                st.rerun()
        
        # Popular Crypto Buttons
        st.write("**Pilihan Crypto Populer (Global)**")
        crypto_cols1 = st.columns(4)
        with crypto_cols1[0]:
            if st.button("BTC-USD", key="btn_btc"):
                st.session_state.custom_stock_input = "BTC-USD"
                st.rerun()
        with crypto_cols1[1]:
            if st.button("ETH-USD", key="btn_eth"):
                st.session_state.custom_stock_input = "ETH-USD"
                st.rerun()
        with crypto_cols1[2]:
            if st.button("SOL-USD", key="btn_sol"):
                st.session_state.custom_stock_input = "SOL-USD"
                st.rerun()
        with crypto_cols1[3]:
            if st.button("BNB-USD", key="btn_bnb"):
                st.session_state.custom_stock_input = "BNB-USD"
                st.rerun()
        
        crypto_cols2 = st.columns(4)
        with crypto_cols2[0]:
            if st.button("XRP-USD", key="btn_xrp"):
                st.session_state.custom_stock_input = "XRP-USD"
                st.rerun()
        with crypto_cols2[1]:
            if st.button("DOGE-USD", key="btn_doge"):
                st.session_state.custom_stock_input = "DOGE-USD"
                st.rerun()
        with crypto_cols2[2]:
            if st.button("ADA-USD", key="btn_ada"):
                st.session_state.custom_stock_input = "ADA-USD"
                st.rerun()
        with crypto_cols2[3]:
            if st.button("DOT-USD", key="btn_dot"):
                st.session_state.custom_stock_input = "DOT-USD"
                st.rerun()
        
        # yFinance Ticker Guidelines Expander (Closed by default)
        with st.expander("💡 Ketentuan & Tips Penulisan Ticker yFinance", expanded=False):
            st.markdown("""
            **Ketentuan Penulisan Ticker yFinance:**
            
            1. **Saham Indonesia (IHSG)**:
               Khusus untuk saham di Bursa Efek Indonesia, tambahkan akhiran **`.JK`** (misalnya: `BBCA.JK`, `BMRI.JK`, `BBNI.JK`, `BRIS.JK`).
               
            2. **Saham Global (AS/S&P 500)**:
               Mengikuti kode ticker standar bursa AS (misalnya: `AAPL`, `GOOGL`, `MSFT`, `TSLA`, `AMZN`). 
               Referensi lengkap dapat dilihat di [Daftar Perusahaan S&P 500](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
               
            3. **Cryptocurrency**:
               Untuk aset crypto, gunakan kode koin diikuti dengan akhiran **`-USD`** (misalnya: `BTC-USD`, `ETH-USD`, `SOL-USD`, `BNB-USD`).
            """)
        
        st.write("")
        st.markdown("**Masukkan Kode Saham/Crypto**")
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            custom_stock = st.text_input(
                "Masukkan Kode Saham", 
                value=st.session_state.custom_stock_input, 
                placeholder="Contoh: BMRI.JK", 
                label_visibility="collapsed"
            )
        with col_btn:
            st.button("Enter 🔍", type="primary", use_container_width=True)
            
        if custom_stock != st.session_state.custom_stock_input:
            st.session_state.custom_stock_input = custom_stock
        
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
