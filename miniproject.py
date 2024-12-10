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



