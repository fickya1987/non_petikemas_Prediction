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
st.image("pelindo_logo.jfif", use_column_width=True)  # Replace with the actual path to the Pelindo logo
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

# Fungsi untuk analisis AI menggunakan GPT-4
def generate_ai_analysis(data, context):
    """
    Generate AI analysis using GPT-4 based on the provided data and context.
    """
    try:
        # Convert data to a summarized string
        data_summary = data.to_string(index=False, max_rows=5)
        messages = [
            {"role": "system", "content": "Anda adalah seorang analis data yang mahir."},
            {"role": "user", "content": f"Berikan analisis naratif berdasarkan data berikut:\n\n{data_summary}\n\n"
                                         f"Konsep: {context}. Tuliskan analisis dengan narasi yang jelas dan terstruktur."}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=2048,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Terjadi kesalahan saat memproses analisis AI: {e}"

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

                        # Tombol untuk Analisis AI
                        if st.button(f"Generate AI Analysis - Dashboard ({category})"):
                            ai_analysis = generate_ai_analysis(data_filtered, f"Dashboard - Analisis Data {category}")
                            st.subheader("Hasil Analisis AI:")
                            st.write(ai_analysis)

# Visualisasi Berdasarkan Kategori
elif menu == "Visualisasi Berdasarkan Kategori":
    st.title(f"Visualisasi Berdasarkan Kategori {category}")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")

            if category == "Petikemas":
                kategori_list = ['JenisPerdagangan', 'JenisKegiatan', 'IsiKargo', 'TipePetiKemas', 'UkuranPetiKemas']
            else:
                kategori_list = ['JenisKargo', 'JenisKemasan', 'JenisKegiatan']

            selected_kategori = st.sidebar.selectbox("Pilih Kategori", kategori_list)

            if selected_kategori in data.columns:
                category_data = data.groupby(selected_kategori)['Value'].sum().reset_index()
                st.subheader(f"Volume Berdasarkan {selected_kategori}")
                fig = px.pie(category_data, values='Value', names=selected_kategori, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                # Tombol untuk Analisis AI
                if st.button(f"Generate AI Analysis - {selected_kategori} ({category})"):
                    ai_analysis = generate_ai_analysis(category_data, f"Visualisasi Berdasarkan {selected_kategori} ({category})")
                    st.subheader("Hasil Analisis AI:")
                    st.write(ai_analysis)

