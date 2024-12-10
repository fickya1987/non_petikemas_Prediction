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
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    data = data.dropna(subset=['Date'])  # Hapus baris dengan tanggal tidak valid
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

            # Periksa apakah kolom 'Date' ada di data
            if 'Date' not in data.columns:
                st.error("Kolom 'Date' tidak ditemukan pada file yang diunggah. Harap pastikan file memiliki kolom 'Date'.")
            else:
                # Konversi kolom 'Date' ke format datetime
                data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
                
                # Hapus baris dengan tanggal tidak valid
                data = data.dropna(subset=['Date'])
                
                if data.empty:
                    st.warning("Data kosong setelah diproses. Harap periksa apakah kolom 'Date' memiliki format yang benar.")
                else:
                    # Tampilkan rentang tanggal yang tersedia
                    min_date = data['Date'].min()
                    max_date = data['Date'].max()
                    st.sidebar.write(f"Rentang Tanggal: {min_date.strftime('%d/%m/%Y')} - {max_date.strftime('%d/%m/%Y')}")

                    # Pilih rentang tanggal
                    start_date = st.sidebar.date_input("Mulai Tanggal", min_date)
                    end_date = st.sidebar.date_input("Akhir Tanggal", max_date)

                    # Filter data berdasarkan rentang tanggal
                    filtered_data = data[(data['Date'] >= pd.Timestamp(start_date)) & (data['Date'] <= pd.Timestamp(end_date))]

                    if filtered_data.empty:
                        st.warning("Tidak ada data yang sesuai dengan rentang tanggal yang dipilih.")
                    else:
                        st.write("Data yang Difilter:")
                        st.write(filtered_data)

                        # Visualisasi nilai total per terminal
                        if 'Terminal' in filtered_data.columns and 'Value' in filtered_data.columns:
                            terminal_volume = filtered_data.groupby('Terminal')['Value'].sum().reset_index()
                            st.subheader("Volume Total Berdasarkan Terminal")
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
            terminal_list = data['Terminal'].unique() if 'Terminal' in data.columns else []
            selected_terminal = st.sidebar.selectbox("Pilih Terminal", terminal_list)

            if selected_terminal:
                terminal_data = data[data['Terminal'] == selected_terminal]
                st.write(f"Data untuk Terminal {selected_terminal}:")
                st.write(terminal_data)

                # Visualisasi data per tanggal
                if 'Value' in terminal_data.columns and 'Date' in terminal_data.columns:
                    terminal_data['Date'] = pd.to_datetime(terminal_data['Date'])
                    volume_by_date = terminal_data.groupby('Date')['Value'].sum().reset_index()
                    st.subheader(f"Volume Harian untuk {selected_terminal}")
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
            selected_category = st.sidebar.selectbox("Pilih Kategori", ['JenisKargo', 'JenisKemasan', 'JenisKegiatan'])
            
            if selected_category in data.columns:
                category_data = data.groupby(selected_category)['Value'].sum().reset_index()
                st.subheader(f"Volume Berdasarkan {selected_category}")
                fig = px.pie(category_data, values='Value', names=selected_category, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)


# Prediction per Terminal
elif menu == "Prediction":
    st.title("Prediksi Data Non Petikemas per Terminal")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")

            # Periksa apakah kolom 'Date' dan 'Terminal' ada di data
            if 'Date' not in data.columns or 'Terminal' not in data.columns:
                st.error("Kolom 'Date' atau 'Terminal' tidak ditemukan pada file yang diunggah. Harap pastikan file memiliki kedua kolom tersebut.")
            else:
                # Konversi kolom 'Date' ke format datetime
                data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
                
                # Hapus baris dengan tanggal tidak valid
                data = data.dropna(subset=['Date'])
                
                if data.empty:
                    st.warning("Data kosong setelah diproses. Harap periksa apakah kolom 'Date' memiliki format yang benar.")
                else:
                    # Pilih terminal untuk dilakukan prediksi
                    terminal_list = data['Terminal'].unique()
                    selected_terminal = st.selectbox("Pilih Terminal", terminal_list)

                    if selected_terminal:
                        terminal_data = data[data['Terminal'] == selected_terminal]
                        terminal_data = terminal_data.sort_values('Date')

                        # Agregasi berdasarkan tanggal
                        data_grouped = terminal_data.groupby('Date')['Value'].sum().reset_index()

                        if data_grouped.empty:
                            st.warning(f"Data kosong untuk terminal '{selected_terminal}'. Tidak ada data untuk diproses.")
                        else:
                            # Pilih periode prediksi
                            forecast_period = st.number_input("Masukkan periode prediksi (dalam bulan)", min_value=1, max_value=24, value=6, step=1)

                            try:
                                # Bangun model SARIMAX
                                model = SARIMAX(data_grouped['Value'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
                                results = model.fit()

                                # Prediksi ke depan
                                future = results.get_forecast(steps=forecast_period)
                                forecast = future.predicted_mean
                                conf_int = future.conf_int()

                                # Validasi indeks terakhir
                                if len(data_grouped) == 0 or pd.isna(data_grouped['Date'].iloc[-1]):
                                    st.warning(f"Tidak dapat menentukan tanggal akhir data untuk prediksi terminal '{selected_terminal}'.")
                                else:
                                    # Tampilkan hasil prediksi
                                    forecast_dates = pd.date_range(start=data_grouped['Date'].iloc[-1], periods=forecast_period + 1, freq='M')[1:]
                                    forecast_df = pd.DataFrame({
                                        'Date': forecast_dates,
                                        'Predicted Value': forecast.values,
                                        'Lower Bound': conf_int.iloc[:, 0].values,
                                        'Upper Bound': conf_int.iloc[:, 1].values
                                    })

                                    st.subheader(f"Hasil Prediksi untuk Terminal '{selected_terminal}'")
                                    st.write(forecast_df)

                                    # Visualisasi hasil prediksi
                                    fig = px.line(forecast_df, x='Date', y='Predicted Value', title=f"Prediksi Nilai Volume Non Petikemas ({selected_terminal})",
                                                  labels={'Predicted Value': 'Value'})
                                    fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Lower Bound'], mode='lines', name='Lower Bound', line=dict(dash='dot'))
                                    fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Upper Bound'], mode='lines', name='Upper Bound', line=dict(dash='dot'))
                                    st.plotly_chart(fig, use_container_width=True)

                                    # Unduh data prediksi
                                    csv = forecast_df.to_csv(index=False).encode('utf-8')
                                    st.download_button(f"Unduh Prediksi untuk Terminal '{selected_terminal}'", data=csv, file_name=f"forecast_{selected_terminal}.csv", mime="text/csv")
                            except Exception as e:
                                st.error(f"Terjadi kesalahan dalam proses prediksi: {e}")
