import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import io

logo_path = "88f2784f-5850-4fbc-8234-0ed3f75fc6b9.png" # Nama file logo hasil upload terbaru

col1, col2 = st.columns([1, 7])
with col1:
    st.image(logo_path, width=90)
with col2:
    st.markdown(
        "<div style='display:flex; align-items:center; height:90px;'>"
        "<span style='font-size:2.15em; font-weight:800; color:#0A3871; line-height:1.13; font-family:Montserrat,Arial,sans-serif;'>"
        "Sistem Prediksi Hasil Panen <br>dengan Whale Optimization + Random Forest"
        "</span></div>",
        unsafe_allow_html=True
    )

st.markdown("""
<style>
.header-row {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 1.5rem;
    margin-bottom: 0.1em;
    margin-top: 1em;
}
.header-logo {
    flex-shrink: 0;
    border-radius: 50%;
    border: 2px solid #0A3871;
    box-shadow: 0 3px 10px rgba(50,50,50,0.09);
    width: 85px;
    height: 85px;
    object-fit: cover;
    background: white;
}
@media (max-width: 600px) {
    .header-row { flex-direction: column; align-items: flex-start; }
    .header-logo { width: 70px; height: 70px;}
}
.header-title {
    font-size: 2.1em !important;
    font-weight: 800 !important;
    color: #12366e;
    line-height: 1.18;
    margin-bottom: 0.10em;
    font-family: 'Montserrat', Arial, sans-serif;
}
.identitas-card {
    background: linear-gradient(90deg, #e6eafc 60%, #d0e0f5 100%);
    border-radius: 13px;
    padding: 20px 30px 16px 28px;
    margin-bottom: 14px;
    margin-top: 0px;
    box-shadow: 0 4px 22px 0 rgba(10,56,113,0.09);
    font-size:1.08em;
}
.identitas-grid {
    display: grid;
    grid-template-columns: 270px 1fr;
    row-gap: 6px;
    align-items: start;
}
.identitas-label {
    font-family: 'Montserrat', Arial, sans-serif;
    color: #143665;
    font-size: 1.09em;
    font-weight: 700;
    letter-spacing: 0.01em;
}
.identitas-value {
    font-family: 'Montserrat', Arial, sans-serif;
    color: #212d4e;
    font-size: 1.09em;
    font-weight: 400;
}

.identitas-card b {
    color: #143665;
    letter-spacing: 0.01em;
}
.identitas-card span {
    color: #212d4e;
}
.petunjuk-card {
    background-color: #f6fafd;
    border-radius: 9px;
    padding: 16px 25px 13px 25px;
    color: #25304f;
    margin-bottom: 17px;
    border-left: 5px solid #3b6ecc;
}
.petunjuk-card ul {
    margin-top:2px; margin-bottom:8px;
    padding-left: 1.2em;
}
.petunjuk-card li { margin-bottom: 2px; }
.prediction-card {
    background: linear-gradient(90deg, #e8f5e8 60%, #d4edda 100%);
    border-radius: 13px;
    padding: 20px 30px 16px 28px;
    margin-bottom: 14px;
    margin-top: 20px;
    box-shadow: 0 4px 22px 0 rgba(40,167,69,0.09);
    border-left: 5px solid #28a745;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="identitas-card">
    <div class="identitas-grid">
        <div class="identitas-label">Nama</div>
        <div class="identitas-value">: Sarmida Uli Sinaga</div>
        <div class="identitas-label">NIM</div>
        <div class="identitas-value">: 211402071</div>
        <div class="identitas-label">Program Studi</div>
        <div class="identitas-value">: Teknologi Informasi</div>
        <div class="identitas-label">Universitas Sumatera Utara</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="petunjuk-card">
    <b>Petunjuk:</b>
    <ul>
        <li>Upload file <b>CSV</b> Anda dengan format yang sesuai (<i>download template</i>).</li>
        <li>Pilih parameter jika diperlukan.</li>
        <li>Setelah data diproses, Anda dapat melihat statistik, visualisasi, dan hasil prediksi.</li>
        <li>Gunakan menu <b>Prediksi Data Baru</b> untuk memprediksi hasil panen pada data baru.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# --- Download Template Data ---
with open("template_data.csv", "rb") as f:
    st.download_button("Download template data CSV", f, file_name="template_data.csv")

st.markdown("---")

# Inisialisasi session state untuk menyimpan model dan scaler
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'final_rf' not in st.session_state:
    st.session_state.final_rf = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None

uploaded_file = st.file_uploader("Upload data panen (CSV):", type="csv")

if uploaded_file:
    # --- Load Data dan Validasi ---
    df = pd.read_csv(uploaded_file)
    st.write("#### Preview Data (5 baris pertama)", df.head())
    st.write(f"**Bentuk data:** {df.shape[0]} baris, {df.shape[1]} kolom")
    
    # Validasi kolom
    required_columns = ['Tahun','Bulan','Luas Panen/ha','Hasil Panen/ton']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Kolom berikut tidak ada: {missing_cols}. Silakan periksa template.")
        st.stop()
    
    # --- Statistik Deskriptif ---
    st.write("#### Statistik Deskriptif")
    st.write(df.describe())

    # --- Distribusi Target ---
    st.write("#### Distribusi Target (Hasil Panen/ton)")
    fig, ax = plt.subplots()
    sns.histplot(df['Hasil Panen/ton'], kde=True, ax=ax, color="skyblue")
    ax.set_xlabel("Hasil Panen/ton")
    st.pyplot(fig)

    # --- Feature Engineering ---
    if 'Bulan' in df.columns:
        df['season'] = df['Bulan'].apply(lambda x: 'dry' if x in [6,7,8,9] else 'wet')
        df['quarter'] = (df['Bulan'] - 1) // 3 + 1
    df = df.sort_values(['Tahun', 'Bulan'])
    df['yield_lag1'] = df['Hasil Panen/ton'].shift(1)
    df['area_lag1'] = df['Luas Panen/ha'].shift(1)
    df = df.dropna()

    # --- Korelasi ---
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    st.write("#### Heatmap Korelasi Fitur")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)

    # --- Pengaturan Split Ratio dan Hyperparameter Advanced ---
    st.markdown("#### Pengaturan Data Split & Hyperparameter")
    split_ratio = st.slider("Pilih rasio data test:", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    with st.expander("Advanced: Pengaturan Whale Optimization (opsional)"):
        n_whales = st.number_input("Jumlah Whale", min_value=5, max_value=30, value=8)
        n_iter = st.number_input("Jumlah Iterasi", min_value=5, max_value=30, value=12)
        n_estimators_range = st.slider("Rentang n_estimators", min_value=50, max_value=500, value=(50,200), step=10)
        max_depth_range = st.slider("Rentang max_depth", min_value=2, max_value=20, value=(3,12), step=1)

    # --- Preprocessing ---
    X = df.drop(['Hasil Panen/ton'], axis=1)
    y = df['Hasil Panen/ton']
    
    # Simpan informasi label encoder untuk prediksi nanti
    le = None
    if 'season' in X.columns:
        le = LabelEncoder()
        X['season_encoded'] = le.fit_transform(X['season'])
        X = X.drop(['season'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Whale Optimization Class ---
    class WhaleOptimization:
        def __init__(self, n_whales, n_iterations, bounds, objective_function):
            self.n_whales = n_whales
            self.n_iterations = n_iterations
            self.bounds = bounds
            self.objective_function = objective_function
            self.dimension = len(bounds)

        def initialize_population(self):
            population = []
            for _ in range(self.n_whales):
                whale = []
                for bound in self.bounds:
                    whale.append(random.uniform(bound[0], bound[1]))
                population.append(whale)
            return np.array(population)

        def optimize(self):
            population = self.initialize_population()
            fitness = [self.objective_function(whale) for whale in population]
            best_idx = np.argmin(fitness)
            best_whale = population[best_idx].copy()
            best_fitness = fitness[best_idx]
            fitness_history = [best_fitness]

            for iteration in range(self.n_iterations):
                a = 2 - iteration * (2 / self.n_iterations)
                for i in range(self.n_whales):
                    r1, r2 = random.random(), random.random()
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    if random.random() < 0.5:
                        if abs(A) < 1:
                            D = abs(C * best_whale - population[i])
                            population[i] = best_whale - A * D
                        else:
                            random_whale = population[random.randint(0, self.n_whales-1)]
                            D = abs(C * random_whale - population[i])
                            population[i] = random_whale - A * D
                    else:
                        distance = abs(best_whale - population[i])
                        b = 1
                        l = random.uniform(-1, 1)
                        population[i] = distance * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale
                    for j in range(self.dimension):
                        population[i][j] = np.clip(population[i][j], self.bounds[j][0], self.bounds[j][1])
                    current_fitness = self.objective_function(population[i])
                    if current_fitness < best_fitness:
                        best_whale = population[i].copy()
                        best_fitness = current_fitness
                fitness_history.append(best_fitness)
            return best_whale, best_fitness, fitness_history

    def calculate_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

    st.markdown("---")
    st.info("Menjalankan Whale Optimization dan Training Model...")
    with st.spinner('Proses tuning dan training model...'):
        def rf_objective_function(params):
            n_estimators = int(params[0])
            max_depth = int(params[1]) if params[1] > 0 else None
            min_samples_split = int(params[2])
            min_samples_leaf = int(params[3])
            max_features = params[4]
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42, n_jobs=-1
            )
            scores = cross_val_score(rf, X_train_scaled, y_train, cv=3, scoring='neg_mean_squared_error')
            return -scores.mean()
        bounds = [
            (n_estimators_range[0], n_estimators_range[1]),   # n_estimators
            (max_depth_range[0], max_depth_range[1]),         # max_depth
            (2, 10),                                          # min_samples_split
            (1, 5),                                           # min_samples_leaf
            (0.3, 1.0)                                        # max_features
        ]
        woa = WhaleOptimization(
            n_whales=int(n_whales),
            n_iterations=int(n_iter),
            bounds=bounds,
            objective_function=rf_objective_function
        )
        best_params, best_fitness, fitness_history = woa.optimize()
        st.success(f"Best params: n_estimators={int(best_params[0])}, max_depth={int(best_params[1])}, "
                   f"min_samples_split={int(best_params[2])}, min_samples_leaf={int(best_params[3])}, "
                   f"max_features={best_params[4]:.2f} (MSE: {best_fitness:.2f})")
        # Log & Grafik
        st.write("##### Grafik Konvergensi Whale Optimization (MSE)")
        fig2, ax2 = plt.subplots()
        ax2.plot(fitness_history, marker='o')
        ax2.set_xlabel("Iterasi")
        ax2.set_ylabel("Best MSE")
        st.pyplot(fig2)

    # --- Training dan Evaluasi Model Akhir (Default/All Features) ---
    final_rf = RandomForestRegressor(
        n_estimators=int(best_params[0]),
        max_depth=int(best_params[1]),
        min_samples_split=int(best_params[2]),
        min_samples_leaf=int(best_params[3]),
        max_features=best_params[4],
        random_state=42, n_jobs=-1
    )
    final_rf.fit(X_train_scaled, y_train)
    y_train_pred = final_rf.predict(X_train_scaled)
    y_test_pred = final_rf.predict(X_test_scaled)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Simpan model dan scaler ke session state
    st.session_state.final_rf = final_rf
    st.session_state.scaler = scaler
    st.session_state.feature_columns = X.columns.tolist()
    st.session_state.label_encoder = le
    st.session_state.model_trained = True
    
    # --- Feature Importance ---
    feat_imp = pd.DataFrame({'feature': X.columns, 'importance': final_rf.feature_importances_}).sort_values('importance', ascending=False)
    st.write("#### Feature Importance")
    st.dataframe(feat_imp)
    
    # --- Tabel Ringkasan Metrik (Model Utama) ---
    st.write("#### Tabel Ringkasan Metrik")
    eval_table = pd.DataFrame([train_metrics, test_metrics], index=['Train', 'Test'])
    st.table(eval_table)
    
    # --- Residual Plot (Model Utama) ---
    st.write("#### Residual Plot (Test Set)")
    fig4, ax4 = plt.subplots()
    residuals = y_test - y_test_pred
    sns.scatterplot(x=range(len(residuals)), y=residuals, ax=ax4)
    ax4.axhline(0, color='r', linestyle='--')
    ax4.set_ylabel("Residual")
    st.pyplot(fig4)
    
    # --- Scatter Actual vs Predicted (Model Utama) ---
    st.write("#### Scatter Actual vs Predicted (Test Set)")
    fig5, ax5 = plt.subplots()
    ax5.scatter(y_test, y_test_pred, color='teal')
    ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax5.set_xlabel("Actual")
    ax5.set_ylabel("Predicted")
    st.pyplot(fig5)
    
    # --- Bar Chart: Actual vs Predicted (Model Utama) ---
    st.write("#### Grafik Bar: Prediksi vs Aktual (Test Set)")
    df_pred = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_test_pred})
    fig6, ax6 = plt.subplots()
    df_pred.head(30).plot(kind='bar', ax=ax6)
    plt.xlabel("Index (sample)")
    plt.ylabel("Hasil Panen/ton")
    st.pyplot(fig6)
    
    # --- Line Chart: Actual vs Predicted (Model Utama) ---
    st.write("#### Line Chart: Predicted vs Actual (Test Set)")
    df_line = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_test_pred}).reset_index(drop=True)
    fig_line, ax_line = plt.subplots(figsize=(10,4))
    ax_line.plot(df_line['Actual'].values, label='Actual', marker='o')
    ax_line.plot(df_line['Predicted'].values, label='Predicted', marker='o')
    ax_line.set_xlabel("Index Data Test")
    ax_line.set_ylabel("Hasil Panen/ton")
    ax_line.legend()
    st.pyplot(fig_line)
    
    # hasil tabel dan downloadable prediction
    test_idx = X_test.index
    kolom_tambahan = ['Tahun', 'Bulan']
    kolom_tambahan = [k for k in kolom_tambahan if k in df.columns]
    df_pred_test = pd.DataFrame({
        **{k: df.loc[test_idx, k].values for k in kolom_tambahan},
        'Actual': y_test.values,
        'Predicted': y_test_pred
    }, index=y_test.index)
    st.write("### Hasil Prediksi pada Data Test Set")
    st.dataframe(df_pred_test.style.format({'Actual':'{:.2f}','Predicted':'{:.2f}'}), height=350)
    csv_pred = df_pred_test.to_csv(index=False).encode()
    st.download_button(
        label="Download hasil prediksi test (CSV)",
        data=csv_pred,
        file_name="hasil_prediksi_test.csv",
        mime='text/csv'
    )

# --- MENU PREDIKSI DATA BARU ---
if st.session_state.model_trained:
    st.markdown("---")
    st.markdown("""
    <div class="prediction-card">
        <h3 style="color: #155724; margin-bottom: 10px;">🔮 Prediksi Data Baru</h3>
        <p style="color: #155724; margin-bottom: 0;">Upload file CSV baru untuk memprediksi hasil panen. File harus berisi kolom: Tahun, Bulan, Luas Panen/ha (tanpa kolom Hasil Panen/ton).</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Template untuk prediksi
    st.markdown("#### Template Data Prediksi")
    template_predict = pd.DataFrame({
        'Tahun': [2024, 2024, 2024],
        'Bulan': [1, 2, 3],
        'Luas Panen/ha': [100, 120, 110]
    })
    st.write("Contoh format data prediksi:")
    st.dataframe(template_predict)
    
    # Download template prediksi
    csv_template = template_predict.to_csv(index=False).encode()
    st.download_button(
        label="Download template prediksi CSV",
        data=csv_template,
        file_name="template_prediksi.csv",
        mime='text/csv'
    )
    
    # Upload file untuk prediksi
    predict_file = st.file_uploader("Upload data untuk prediksi (CSV):", type="csv", key="predict_file")
    
    if predict_file:
        try:
            # Load data prediksi
            df_predict = pd.read_csv(predict_file)
            st.write("#### Preview Data Prediksi")
            st.dataframe(df_predict)
            
            # Validasi kolom yang diperlukan
            required_predict_cols = ['Tahun', 'Bulan', 'Luas Panen/ha']
            missing_predict_cols = [col for col in required_predict_cols if col not in df_predict.columns]
            
            if missing_predict_cols:
                st.error(f"Kolom berikut tidak ada dalam data prediksi: {missing_predict_cols}")
            else:
                # Proses feature engineering sama seperti data training
                if 'Bulan' in df_predict.columns:
                    df_predict['season'] = df_predict['Bulan'].apply(lambda x: 'dry' if x in [6,7,8,9] else 'wet')
                    df_predict['quarter'] = (df_predict['Bulan'] - 1) // 3 + 1
                
                # Untuk lag features, kita bisa menggunakan nilai rata-rata atau nilai terakhir dari data training
                # Di sini kita akan menggunakan strategi sederhana dengan mengisi nilai rata-rata
                df_predict['yield_lag1'] = df['Hasil Panen/ton'].mean()  # Menggunakan rata-rata hasil panen dari data training
                df_predict['area_lag1'] = df_predict['Luas Panen/ha'].shift(1)
                df_predict['area_lag1'] = df_predict['area_lag1'].fillna(df_predict['Luas Panen/ha'].mean())
                
                # Encode season jika ada
                if 'season' in df_predict.columns and st.session_state.label_encoder:
                    df_predict['season_encoded'] = st.session_state.label_encoder.transform(df_predict['season'])
                    df_predict = df_predict.drop(['season'], axis=1)
                
                # Pastikan kolom sesuai dengan yang digunakan saat training
                X_predict = df_predict[st.session_state.feature_columns]
                
                # Scaling
                X_predict_scaled = st.session_state.scaler.transform(X_predict)
                
                # Prediksi
                predictions = st.session_state.final_rf.predict(X_predict_scaled)
                
                # Tampilkan hasil prediksi
                st.write("#### Hasil Prediksi")
                df_result = df_predict[['Tahun', 'Bulan', 'Luas Panen/ha']].copy()
                df_result['Prediksi Hasil Panen/ton'] = predictions
                
                st.dataframe(df_result.style.format({'Prediksi Hasil Panen/ton': '{:.2f}'}))
                
                # Visualisasi hasil prediksi
                st.write("#### Visualisasi Prediksi")
                fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
                
                # Bar chart prediksi
                x_pos = range(len(df_result))
                bars = ax_pred.bar(x_pos, predictions, color='lightgreen', alpha=0.7)
                ax_pred.set_xlabel('Index Data')
                ax_pred.set_ylabel('Prediksi Hasil Panen/ton')
                ax_pred.set_title('Prediksi Hasil Panen')
                
                # Tambahkan label pada bar
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax_pred.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}',
                               ha='center', va='bottom')
                
                st.pyplot(fig_pred)
                
                # Line chart jika ada lebih dari 1 data
                if len(df_result) > 1:
                    fig_line_pred, ax_line_pred = plt.subplots(figsize=(10, 5))
                    ax_line_pred.plot(predictions, marker='o', color='green', linewidth=2)
                    ax_line_pred.set_xlabel('Index Data')
                    ax_line_pred.set_ylabel('Prediksi Hasil Panen/ton')
                    ax_line_pred.set_title('Trend Prediksi Hasil Panen')
                    ax_line_pred.grid(True, alpha=0.3)
                    st.pyplot(fig_line_pred)
                
                # Statistik prediksi
                st.write("#### Statistik Prediksi")
                pred_stats = pd.DataFrame({
                    'Metrik': ['Rata-rata', 'Median', 'Minimum', 'Maksimum', 'Std Deviasi'],
                    'Nilai': [
                        predictions.mean(),
                        np.median(predictions),
                        predictions.min(),
                        predictions.max(),
                        predictions.std()
                    ]
                })
                st.dataframe(pred_stats.style.format({'Nilai': '{:.2f}'}))
                
                # Download hasil prediksi
                csv_result = df_result.to_csv(index=False).encode()
                st.download_button(
                    label="Download hasil prediksi (CSV)",
                    data=csv_result,
                    file_name="hasil_prediksi_baru.csv",
                    mime='text/csv'
                )
                
                st.success(f"Prediksi berhasil! {len(df_result)} data telah diprediksi.")
                
        except Exception as e:
            st.error(f"Terjadi error saat memproses data prediksi: {str(e)}")
            st.write("Pastikan format data sesuai dengan template yang disediakan.")

else:
    st.markdown("---")
    st.info("⚠️ Silakan upload dan proses data training terlebih dahulu untuk mengaktifkan fitur prediksi.")
    
# --- Tentang & Referensi ---
st.markdown("""
---
<b>Tentang:</b> Sistem ini dikembangkan untuk mendemonstrasikan integrasi Whale Optimization Algorithm (WOA) dengan Random Forest dalam prediksi hasil panen berbasis data.
""", unsafe_allow_html=True)
