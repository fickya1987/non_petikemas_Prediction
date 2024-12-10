import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

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

# Fungsi utama untuk memfilter dan menampilkan data
def filter_data(data):
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    min_date = data['Date'].min()
    max_date = data['Date'].max()
    
    start_date = st.sidebar.date_input("Mulai Tanggal", min_date)
    end_date = st.sidebar.date_input("Akhir Tanggal", max_date)
    
    return data[(data['Date'] >= pd.Timestamp(start_date)) & (data['Date'] <= pd.Timestamp(end_date))]

# Dashboard
if menu == "Dashboard":
    st.title("Analisis Data Non Petikemas")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")
            filtered_data = filter_data(data)
            st.write(filtered_data)

            # Visualisasi nilai total per terminal
            if 'Terminal' in filtered_data.columns and 'Value' in filtered_data.columns:
                terminal_volume = filtered_data.groupby('Terminal')['Value'].sum().reset_index()
                st.subheader("Volume Total Berdasarkan Terminal")
                fig = px.bar(terminal_volume, x='Terminal', y='Value', text='Value', template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

# Prediction
elif menu == "Prediction":
    st.title("Prediksi Data Non Petikemas")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")
            
            # Konversi kolom 'Date' ke format datetime
            data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
            data = data.sort_values('Date')

            # Agregasi berdasarkan tanggal (jika ada data duplikat per tanggal)
            data_grouped = data.groupby('Date')['Value'].sum().reset_index()

            # Pilih periode prediksi
            forecast_period = st.number_input("Masukkan periode prediksi (dalam bulan)", min_value=1, max_value=24, value=6, step=1)

            # Bangun model SARIMAX
            model = SARIMAX(data_grouped['Value'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit()

            # Prediksi ke depan
            future = results.get_forecast(steps=forecast_period)
            forecast = future.predicted_mean
            conf_int = future.conf_int()

            # Tampilkan hasil prediksi
            forecast_dates = pd.date_range(start=data_grouped['Date'].iloc[-1], periods=forecast_period + 1, freq='M')[1:]
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Predicted Value': forecast.values,
                'Lower Bound': conf_int.iloc[:, 0].values,
                'Upper Bound': conf_int.iloc[:, 1].values
            })

            st.subheader("Hasil Prediksi")
            st.write(forecast_df)

            # Visualisasi hasil prediksi
            fig = px.line(forecast_df, x='Date', y='Predicted Value', title="Prediksi Nilai Volume Non Petikemas",
                          labels={'Predicted Value': 'Value'})
            fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Lower Bound'], mode='lines', name='Lower Bound', line=dict(dash='dot'))
            fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Upper Bound'], mode='lines', name='Upper Bound', line=dict(dash='dot'))
            st.plotly_chart(fig, use_container_width=True)

            # Unduh data prediksi
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("Unduh Prediksi", data=csv, file_name="forecast_prediction.csv", mime="text/csv")
