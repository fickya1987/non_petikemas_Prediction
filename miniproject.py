import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import subprocess
import os
import warnings

# Install required libraries if not already installed
try:
    import openai
except ModuleNotFoundError:
    subprocess.check_call(["pip", "install", "openai"])
    import openai

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    subprocess.check_call(["pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Check if API key exists
if not openai.api_key:
    raise ValueError("API Key OpenAI tidak ditemukan. Harap tambahkan ke file .env.")

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Non Petikemas Analysis", layout="wide")

# Sidebar
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Pilih Menu", ["Dashboard", "Analisis Terminal", "Visualisasi Berdasarkan Kategori", "Prediction"])

# Fungsi untuk membaca data berdasarkan tipe file
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Format file tidak didukung. Harap unggah file .csv atau .xlsx.")
        return None

# Fungsi untuk analisis AI menggunakan GPT-4
def generate_ai_analysis(data, context):
    """
    Generate AI analysis using GPT-4 based on the provided data and context.
    """
    try:
        # Convert data to a summarized string
        data_summary = data.to_string(index=False, max_rows=5)  # Show only top 5 rows
        messages = [
            {"role": "system", "content": "Anda adalah seorang analis data yang mahir."},
            {"role": "user", "content": f"Berikan analisis naratif berdasarkan data berikut:\n\n{data_summary}\n\n"
                                         f"Konsep: {context}. Tuliskan analisis dengan narasi yang jelas dan terstruktur."}
        ]
        # Call OpenAI GPT-4 API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=2048,
            temperature=1.0
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Terjadi kesalahan saat memproses analisis AI: {e}"

# Fungsi utama untuk memfilter dan menampilkan data
def filter_data(data):
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
    data = data.dropna(subset=['Date'])  # Hapus baris dengan tanggal tidak valid
    return data

# Dashboard
if menu == "Dashboard":
    st.title("Analisis Data Non Petikemas")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")

            if 'Date' not in data.columns or 'Satuan' not in data.columns:
                st.error("Kolom 'Date' atau 'Satuan' tidak ditemukan pada file yang diunggah.")
            else:
                data = filter_data(data)
                
                if data.empty:
                    st.warning("Data kosong setelah diproses. Harap periksa format data.")
                else:
                    satuan_list = data['Satuan'].unique()
                    selected_satuan = st.sidebar.selectbox("Pilih Satuan", satuan_list)

                    if selected_satuan:
                        data_filtered = data[data['Satuan'] == selected_satuan]

                        min_date = data_filtered['Date'].min()
                        max_date = data_filtered['Date'].max()
                        st.sidebar.write(f"Rentang Tanggal: {min_date.strftime('%d/%m/%Y')} - {max_date.strftime('%d/%m/%Y')}")

                        start_date = st.sidebar.date_input("Mulai Tanggal", min_date)
                        end_date = st.sidebar.date_input("Akhir Tanggal", max_date)

                        data_filtered = data_filtered[(data_filtered['Date'] >= pd.Timestamp(start_date)) & 
                                                       (data_filtered['Date'] <= pd.Timestamp(end_date))]

                        if data_filtered.empty:
                            st.warning("Tidak ada data yang sesuai dengan rentang tanggal dan satuan yang dipilih.")
                        else:
                            st.write(f"Data yang Difilter (Satuan: {selected_satuan}):")
                            st.write(data_filtered)

                            # Tombol untuk Analisis AI
                            if st.button("Generate AI Analysis - Dashboard"):
                                ai_analysis = generate_ai_analysis(data_filtered, "Dashboard - Analisis Data Non Petikemas")
                                st.subheader("Hasil Analisis AI:")
                                st.write(ai_analysis)

# Analisis Terminal
elif menu == "Analisis Terminal":
    st.title("Analisis Terminal Non Petikemas")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")

            if 'Satuan' not in data.columns or 'Terminal' not in data.columns:
                st.error("Kolom 'Satuan' atau 'Terminal' tidak ditemukan pada file yang diunggah.")
            else:
                data = filter_data(data)

                if data.empty:
                    st.warning("Data kosong setelah diproses. Harap periksa format data.")
                else:
                    satuan_list = data['Satuan'].unique()
                    selected_satuan = st.sidebar.selectbox("Pilih Satuan", satuan_list)

                    if selected_satuan:
                        data_filtered = data[data['Satuan'] == selected_satuan]
                        terminal_list = data_filtered['Terminal'].unique()
                        selected_terminal = st.sidebar.selectbox("Pilih Terminal", terminal_list)

                        if selected_terminal:
                            terminal_data = data_filtered[data_filtered['Terminal'] == selected_terminal]

                            if terminal_data.empty:
                                st.warning(f"Tidak ada data untuk terminal '{selected_terminal}' dengan satuan '{selected_satuan}'.")
                            else:
                                st.write(f"Data untuk Terminal '{selected_terminal}' (Satuan: {selected_satuan}):")
                                st.write(terminal_data)

                                # Tombol untuk Analisis AI
                                if st.button("Generate AI Analysis - Terminal"):
                                    ai_analysis = generate_ai_analysis(terminal_data, f"Analisis Terminal {selected_terminal}")
                                    st.subheader("Hasil Analisis AI:")
                                    st.write(ai_analysis)

# Visualisasi Berdasarkan Kategori
elif menu == "Visualisasi Berdasarkan Kategori":
    st.title("Visualisasi Berdasarkan Kategori")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")

            if 'Satuan' not in data.columns:
                st.error("Kolom 'Satuan' tidak ditemukan pada file yang diunggah.")
            else:
                data = filter_data(data)

                if data.empty:
                    st.warning("Data kosong setelah diproses. Harap periksa format data.")
                else:
                    satuan_list = data['Satuan'].unique()
                    selected_satuan = st.sidebar.selectbox("Pilih Satuan", satuan_list)

                    if selected_satuan:
                        data_filtered = data[data['Satuan'] == selected_satuan]
                        kategori_list = ['JenisKargo', 'JenisKemasan', 'JenisKegiatan']
                        selected_kategori = st.sidebar.selectbox("Pilih Kategori", kategori_list)

                        if selected_kategori in data_filtered.columns:
                            category_data = data_filtered.groupby(selected_kategori)['Value'].sum().reset_index()
                            st.subheader(f"Volume Berdasarkan {selected_kategori} (Satuan: {selected_satuan})")
                            fig = px.pie(category_data, values='Value', names=selected_kategori, template='plotly_white')
                            st.plotly_chart(fig, use_container_width=True)

                            # Tombol untuk Analisis AI
                            if st.button(f"Generate AI Analysis - {selected_kategori}"):
                                ai_analysis = generate_ai_analysis(category_data, f"Visualisasi Berdasarkan {selected_kategori}")
                                st.subheader("Hasil Analisis AI:")
                                st.write(ai_analysis)

# Prediction
elif menu == "Prediction":
    st.title("Prediksi Data Non Petikemas per Terminal dan Satuan")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")

            # Validasi kolom wajib
            required_columns = ['Date', 'Terminal', 'Satuan', 'Value']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing_columns)}")
            else:
                # Konversi 'Date' ke format datetime
                data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
                data = data.dropna(subset=['Date'])

                if data.empty:
                    st.warning("Data kosong setelah diproses. Periksa format file yang diunggah.")
                else:
                    # Filter berdasarkan satuan
                    satuan_list = data['Satuan'].unique()
                    selected_satuan = st.sidebar.selectbox("Pilih Satuan", satuan_list)

                    if selected_satuan:
                        data_satuan = data[data['Satuan'] == selected_satuan]

                        # Filter berdasarkan terminal
                        terminal_list = data_satuan['Terminal'].unique()
                        selected_terminal = st.selectbox("Pilih Terminal", terminal_list)

                        if selected_terminal:
                            terminal_data = data_satuan[data_satuan['Terminal'] == selected_terminal]
                            terminal_data = terminal_data.sort_values('Date')

                            # Agregasi berdasarkan tanggal
                            data_grouped = terminal_data.groupby('Date')['Value'].sum().reset_index()

                            if data_grouped.empty:
                                st.warning(f"Data kosong untuk terminal '{selected_terminal}' dengan satuan '{selected_satuan}'.")
                            else:
                                # Validasi jumlah data untuk prediksi
                                if len(data_grouped) < 12:
                                    st.error("Data terlalu sedikit untuk prediksi. Tambahkan data dengan rentang waktu yang lebih panjang.")
                                else:
                                    forecast_period = st.number_input("Masukkan periode prediksi (dalam bulan)", min_value=1, max_value=24, value=6, step=1)

                                    try:
                                        # Bangun model SARIMAX
                                        model = SARIMAX(data_grouped['Value'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
                                        results = model.fit()

                                        # Prediksi ke depan
                                        future = results.get_forecast(steps=forecast_period)
                                        forecast = future.predicted_mean
                                        conf_int = future.conf_int()

                                        forecast_dates = pd.date_range(start=data_grouped['Date'].iloc[-1], periods=forecast_period + 1, freq='M')[1:]
                                        forecast_df = pd.DataFrame({
                                            'Date': forecast_dates,
                                            'Predicted Value': forecast.values,
                                            'Lower Bound': conf_int.iloc[:, 0].values,
                                            'Upper Bound': conf_int.iloc[:, 1].values
                                        })

                                        st.subheader(f"Hasil Prediksi untuk Terminal '{selected_terminal}' (Satuan: {selected_satuan})")
                                        st.write(forecast_df)

                                        # Visualisasi hasil prediksi
                                        fig = px.line(forecast_df, x='Date', y='Predicted Value', title=f"Prediksi ({selected_terminal}, {selected_satuan})")
                                        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Lower Bound'], mode='lines', name='Lower Bound', line=dict(dash='dot'))
                                        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Upper Bound'], mode='lines', name='Upper Bound', line=dict(dash='dot'))
                                        st.plotly_chart(fig, use_container_width=True)

                                        # Tombol untuk Analisis AI
                                        if st.button("Generate AI Analysis - Prediction"):
                                            ai_analysis = generate_ai_analysis(forecast_df, f"Prediksi untuk Terminal {selected_terminal} (Satuan: {selected_satuan})")
                                            st.subheader("Hasil Analisis AI:")
                                            st.write(ai_analysis)
                                    except Exception as e:
                                        st.error(f"Terjadi kesalahan dalam proses prediksi: {e}")



def preprocess_data(data):
    try:
        # Debug awal: Menampilkan data asli
        st.write("Data awal yang diterima:")
        st.write(data.head())

        # Pastikan kolom 'Date' ada
        if 'Date' not in data.columns:
            raise ValueError("Kolom 'Date' tidak ditemukan pada data.")

        # Debug sebelum konversi
        st.write("Sebelum konversi 'Date':")
        st.write(data['Date'].head())

        # Konversi kolom 'Date' ke format datetime
        data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M', errors='coerce')

        # Debug setelah konversi
        st.write("Setelah konversi 'Date':")
        st.write(data.head())

        # Hapus baris dengan tanggal tidak valid
        data = data.dropna(subset=['Date'])

        # Debug setelah drop NA
        st.write("Setelah menghapus baris dengan 'Date' tidak valid:")
        st.write(data.head())

        # Isi nilai NaN di kolom lain dengan 0
        data.fillna(0, inplace=True)

        # Debug setelah pengisian NaN
        st.write("Setelah mengisi nilai kosong dengan 0:")
        st.write(data.head())

        return data
    except Exception as e:
        st.error(f"Kesalahan saat preprocessing data: {e}")
        return None


# Preprocessing

elif menu == "Preprocessing":
    st.title("Preprocessing Data Petikemas")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data asli:")
            st.write(data.head())

            try:
                preprocessed_data = preprocess_data(data)
                if preprocessed_data is not None:
                    st.write("Data setelah preprocessing:")
                    st.write(preprocessed_data.head())

                    # Unduh data hasil preprocessing
                    csv = preprocessed_data.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Unduh Data Preprocessed", data=csv, file_name="preprocessed_data.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Kesalahan selama preprocessing: {e}")



