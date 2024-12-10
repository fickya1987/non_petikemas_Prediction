# Prediction
elif menu == "Prediction":
    st.title("Prediksi Data Non Petikemas")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")
            
            # Konversi kolom 'Date' ke format datetime
            data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
            data = data.dropna(subset=['Date'])  # Hapus baris dengan tanggal tidak valid
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

