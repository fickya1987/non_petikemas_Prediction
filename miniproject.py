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

st.set_page_config(page_title="Petikemas dan Non-Petikemas Analysis", layout="wide")

# Sidebar
st.sidebar.title("Menu")
category = st.sidebar.radio("Pilih Kategori", ["Petikemas", "Non-Petikemas"])
menu = st.sidebar.selectbox("Pilih Menu", ["Dashboard", "Prediction"])

# Load data based on category
if category == "Petikemas":
    file_path = "adjusted_petikemas_priok_data.csv"
else:
    file_path = "adjusted_non_petikemas_priok_data.csv"

# Load the data
try:
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
    data = data.dropna(subset=['Date'])  # Drop rows with invalid dates
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data: {e}")
    data = None

# Dashboard
if menu == "Dashboard" and data is not None:
    st.title(f"Analisis Data {category}")
    st.write("Data berhasil dimuat!")

    if 'Satuan' not in data.columns:
        st.error("Kolom 'Satuan' tidak ditemukan pada file.")
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

# Prediction
elif menu == "Prediction" and data is not None:
    st.title(f"Prediksi Data {category}")
    st.write("Data berhasil dimuat!")

    required_columns = ['Date', 'Satuan', 'Value']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing_columns)}")
    else:
        satuan_list = data['Satuan'].unique()
        selected_satuan = st.sidebar.selectbox("Pilih Satuan", satuan_list)

        if selected_satuan:
            data_satuan = data[data['Satuan'] == selected_satuan]
            data_satuan = data_satuan.sort_values('Date')

            data_grouped = data_satuan.groupby('Date')['Value'].sum().reset_index()

            if data_grouped.empty:
                st.warning("Data kosong setelah diproses. Harap periksa file.")
            else:
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
