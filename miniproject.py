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

# Set Streamlit page configuration
st.set_page_config(page_title="Analytical Dashboard TKMP Pelindo", layout="wide")

# Header with logo and title
st.image("pelindo_logo.png", use_column_width=True)  # Replace with the actual path to the Pelindo logo
st.title("Analytical Dashboard TKMP Pelindo")
st.markdown("---")

# Sidebar
st.sidebar.title("Menu")
category = st.sidebar.radio("Pilih Kategori", ["Petikemas", "Non-Petikemas"])
menu = st.sidebar.selectbox("Pilih Menu", ["Dashboard", "Analisis Terminal", "Visualisasi Berdasarkan Kategori", "Prediction"])

# Fungsi untuk membaca data dari file
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Format file tidak didukung. Harap unggah file .csv atau .xlsx.")
        return None

# Dashboard
if menu == "Dashboard":
    st.title(f"Dashboard Data {category}")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")

            if 'Date' not in data.columns or 'Terminal' not in data.columns or 'Satuan' not in data.columns:
                st.error("Kolom 'Date', 'Terminal', atau 'Satuan' tidak ditemukan pada file yang diunggah.")
            else:
                data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
                data = data.dropna(subset=['Date'])

                # Pilihan terminal
                terminal_list = data['Terminal'].unique()
                selected_terminal = st.sidebar.selectbox("Pilih Terminal", terminal_list)

                if selected_terminal:
                    data_filtered = data[data['Terminal'] == selected_terminal]

                    # Pilihan satuan
                    satuan_list = data_filtered['Satuan'].unique()
                    selected_satuan = st.sidebar.selectbox("Pilih Satuan", satuan_list)

                    if selected_satuan:
                        data_filtered = data_filtered[data_filtered['Satuan'] == selected_satuan]
                        st.write(f"Data Terminal {selected_terminal} (Satuan: {selected_satuan})")
                        st.write(data_filtered)

# Analisis Terminal
elif menu == "Analisis Terminal":
    st.title(f"Analisis Terminal {category}")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")

            if 'Terminal' not in data.columns or 'Value' not in data.columns:
                st.error("Kolom 'Terminal' atau 'Value' tidak ditemukan pada file yang diunggah.")
            else:
                terminal_list = data['Terminal'].unique()
                selected_terminal = st.sidebar.selectbox("Pilih Terminal", terminal_list)

                if selected_terminal:
                    terminal_data = data[data['Terminal'] == selected_terminal]
                    st.write(f"Data untuk Terminal '{selected_terminal}':")
                    st.write(terminal_data)

# Visualisasi Berdasarkan Kategori
elif menu == "Visualisasi Berdasarkan Kategori":
    st.title(f"Visualisasi Berdasarkan Kategori {category}")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")

            if 'Terminal' not in data.columns or 'Value' not in data.columns:
                st.error("Kolom 'Terminal' atau 'Value' tidak ditemukan pada file yang diunggah.")
            else:
                terminal_list = data['Terminal'].unique()
                selected_terminal = st.sidebar.selectbox("Pilih Terminal", terminal_list)

                if selected_terminal:
                    terminal_data = data[data['Terminal'] == selected_terminal]
                    kategori_list = ['JenisKargo', 'JenisKemasan', 'JenisKegiatan']
                    selected_kategori = st.sidebar.selectbox("Pilih Kategori", kategori_list)

                    if selected_kategori in terminal_data.columns:
                        category_data = terminal_data.groupby(selected_kategori)['Value'].sum().reset_index()
                        st.subheader(f"Volume Berdasarkan {selected_kategori}")
                        fig = px.pie(category_data, values='Value', names=selected_kategori, template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)

# Prediction
elif menu == "Prediction":
    st.title(f"Prediksi Data {category}")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")

            if 'Date' not in data.columns or 'Value' not in data.columns:
                st.error("Kolom 'Date' atau 'Value' tidak ditemukan pada file yang diunggah.")
            else:
                data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
                data_grouped = data.groupby('Date')['Value'].sum().reset_index()

                if len(data_grouped) < 12:
                    st.error("Data terlalu sedikit untuk prediksi. Tambahkan data dengan rentang waktu yang lebih panjang.")
                else:
                    forecast_period = st.number_input("Masukkan periode prediksi (dalam bulan)", min_value=1, max_value=24, value=6, step=1)

                    try:
                        model = SARIMAX(data_grouped['Value'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                                        enforce_stationarity=False, enforce_invertibility=False)
                        results = model.fit()

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

                        st.subheader("Hasil Prediksi")
                        st.write(forecast_df)

                        fig = px.line(forecast_df, x='Date', y='Predicted Value', title=f"Prediksi Data {category}")
                        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Lower Bound'], mode='lines', name='Lower Bound', line=dict(dash='dot'))
                        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Upper Bound'], mode='lines', name='Upper Bound', line=dict(dash='dot'))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Terjadi kesalahan dalam proses prediksi: {e}")

