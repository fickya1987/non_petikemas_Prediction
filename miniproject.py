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

                            if 'Terminal' in data_filtered.columns and 'Value' in data_filtered.columns:
                                terminal_volume = data_filtered.groupby('Terminal')['Value'].sum().reset_index()
                                st.subheader(f"Volume Total Berdasarkan Terminal (Satuan: {selected_satuan})")
                                fig = px.bar(terminal_volume, x='Terminal', y='Value', text='Value', template='plotly_white')
                                st.plotly_chart(fig, use_container_width=True)

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

                                if 'Value' in terminal_data.columns and 'Date' in terminal_data.columns:
                                    volume_by_date = terminal_data.groupby('Date')['Value'].sum().reset_index()
                                    st.subheader(f"Volume Harian untuk Terminal '{selected_terminal}' (Satuan: {selected_satuan})")
                                    fig = px.line(volume_by_date, x='Date', y='Value', markers=True, template='plotly_white')
                                    st.plotly_chart(fig, use_container_width=True)

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

# Prediction
elif menu == "Prediction":
    st.title("Prediksi Data Non Petikemas per Terminal dan Satuan")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")

            if 'Date' not in data.columns or 'Terminal' not in data.columns or 'Satuan' not in data.columns:
                st.error("Kolom 'Date', 'Terminal', atau 'Satuan' tidak ditemukan pada file yang diunggah.")
            else:
                data = filter_data(data)

                if data.empty:
                    st.warning("Data kosong setelah diproses. Harap periksa format data.")
                else:
                    satuan_list = data['Satuan'].unique()
                    selected_satuan = st.sidebar.selectbox("Pilih Satuan", satuan_list)

                    if selected_satuan:
                        data_satuan = data[data['Satuan'] == selected_satuan]
                        terminal_list = data_satuan['Terminal'].unique()
                        selected_terminal = st.selectbox("Pilih Terminal", terminal_list)

                        if selected_terminal:
                            terminal_data = data_satuan[data_satuan['Terminal'] == selected_terminal]

                            if terminal_data.empty:
                                st.warning(f"Data kosong untuk terminal '{selected_terminal}' dengan satuan '{selected_satuan}'.")
                            else:
                                data_grouped = terminal_data.groupby('Date')['Value'].sum().reset_index()

                                forecast_period = st.number_input("Masukkan periode prediksi (dalam bulan)", min_value=1, max_value=24, value=6, step=1)

                                try:
                                    model = SARIMAX(data_grouped['Value'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
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

                                    st.subheader(f"Hasil Prediksi untuk Terminal '{selected_terminal}' (Satuan: {selected_satuan})")
                                    st.write(forecast_df)

                                    fig = px.line(forecast_df, x='Date', y='Predicted Value', title=f"Prediksi ({selected_terminal}, {selected_satuan})")
                                    fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Lower Bound'], mode='lines', name='Lower Bound', line=dict(dash='dot'))
                                    fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Upper Bound'], mode='lines', name='Upper Bound', line=dict(dash='dot'))
                                    st.plotly_chart(fig, use_container_width=True)

                                    csv = forecast_df.to_csv(index=False).encode('utf-8')
                                    st.download_button(f"Unduh Prediksi Terminal '{selected_terminal}' dan Satuan '{selected_satuan}'", data=csv, file_name=f"forecast_{selected_terminal}_{selected_satuan}.csv", mime="text/csv")
                                except Exception as e:
                                    st.error(f"Terjadi kesalahan dalam proses prediksi: {e}")

