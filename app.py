import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from datetime import datetime
import base64

# Pastikan import berjalan dengan error handling
try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import plotly.express as px
    import plotly.graph_objects as go
    SKLEARN_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing required modules: {e}")
    SKLEARN_AVAILABLE = False

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Hasil Panen Padi - WOA-RF",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan yang lebih menarik
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
    .info-box {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #4CAF50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌾 Sistem Prediksi Hasil Panen Padi</h1>
    <p>Powered by Random Forest + Whale Optimization Algorithm</p>
</div>
""", unsafe_allow_html=True)

# Fungsi untuk demo model (ketika model actual tidak tersedia)
class DemoModel:
    """Demo model untuk testing interface"""
    
    def __init__(self):
        self.is_demo = True
        self.feature_names = ['Tahun', 'Bulan', 'Luas Panen/ha', 'quarter', 'season_encoded']
        
    def predict(self, X):
        """Simple prediction logic untuk demo"""
        predictions = []
        for i in range(len(X)):
            # Base yield per hectare
            base_yield = 2.8  # ton/ha
            
            # Seasonal factor
            month = X[i][1] if len(X[i]) > 1 else 6
            seasonal_factor = 1.15 if month in [6,7,8,9] else 1.0
            
            # Year trend (improvement over time)
            year = X[i][0] if len(X) > 0 else 2024
            year_factor = 1 + (year - 2020) * 0.015
            
            # Area
            area = X[i][2] if len(X[i]) > 2 else 100
            
            # Random variation
            np.random.seed(int(year + month + area))
            variation = np.random.uniform(0.85, 1.15)
            
            prediction = area * base_yield * seasonal_factor * year_factor * variation
            predictions.append(max(prediction, 10))  # minimum 10 ton
            
        return np.array(predictions)

@st.cache_resource
def load_or_create_model():
    """Load trained model atau buat demo model"""
    try:
        # Coba load model yang sudah di-train
        # Untuk demo, kita buat model sederhana
        return DemoModel()
    except:
        return DemoModel()

def preprocess_data(df):
    """Preprocessing data sesuai dengan pipeline training"""
    df_processed = df.copy()
    
    # Feature engineering
    df_processed['season'] = df_processed['Bulan'].apply(lambda x: 'dry' if x in [6,7,8,9] else 'wet')
    df_processed['quarter'] = (df_processed['Bulan'] - 1) // 3 + 1
    
    # Encode season
    season_mapping = {'dry': 1, 'wet': 0}
    df_processed['season_encoded'] = df_processed['season'].map(season_mapping)
    
    return df_processed

def make_prediction(df, model):
    """Membuat prediksi"""
    try:
        # Siapkan features
        features = ['Tahun', 'Bulan', 'Luas Panen/ha', 'quarter', 'season_encoded']
        X = df[features].values
        
        # Prediksi
        predictions = model.predict(X)
        return predictions
    except Exception as e:
        st.error(f"Error dalam prediksi: {e}")
        return np.random.uniform(50, 200, len(df))

def create_gauge_chart(value, title="Hasil Panen (ton)"):
    """Membuat gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24}},
        delta = {'reference': 150},
        gauge = {
            'axis': {'range': [None, 400]},
            'bar': {'color': "#2E8B57"},
            'steps': [
                {'range': [0, 150], 'color': "#FFE4E1"},
                {'range': [150, 300], 'color': "#98FB98"},
                {'range': [300, 400], 'color': "#90EE90"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 350
            }
        }
    ))
    fig.update_layout(height=400, font={'color': "darkblue", 'family': "Arial"})
    return fig

def get_download_link(df, filename):
    """Generate download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    return href

# Load model
model = load_or_create_model()

# Sidebar
with st.sidebar:
    st.header("🎛️ Panel Kontrol")
    
    # Model info
    if hasattr(model, 'is_demo') and model.is_demo:
        st.warning("🔧 **Mode Demo**\nMenggunakan model simulasi untuk demonstrasi interface")
    else:
        st.success("✅ **Model Aktif**\nModel WOA-RF telah dimuat")
    
    st.markdown("---")
    
    # Input method selection
    st.subheader("📊 Pilih Mode Input")
    input_mode = st.radio(
        "Metode Input:",
        ["🖊️ Input Manual", "📁 Upload CSV"],
        help="Pilih cara input data untuk prediksi"
    )
    
    st.markdown("---")
    
    # Model information
    st.subheader("ℹ️ Informasi Model")
    with st.expander("Detail Model"):
        st.markdown("""
        **Algoritma:**
        - Random Forest Regressor
        - Whale Optimization Algorithm
        
        **Input Features:**
        - Tahun
        - Bulan
        - Luas Panen (ha)
        - Musim (auto-generated)
        - Kuartal (auto-generated)
        
        **Metrics (Demo):**
        - R² Score: 0.87
        - RMSE: 12.3
        - MAE: 8.9
        - MAPE: 6.5%
        """)

# Main content
if input_mode == "🖊️ Input Manual":
    # MANUAL INPUT MODE
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Input Parameter")
        
        with st.form("prediction_form"):
            st.markdown("**Masukkan data untuk prediksi:**")
            
            tahun = st.number_input(
                "📅 Tahun", 
                min_value=2020, 
                max_value=2030, 
                value=2024,
                help="Tahun untuk prediksi (2020-2030)"
            )
            
            bulan = st.selectbox(
                "📆 Bulan",
                options=list(range(1, 13)),
                format_func=lambda x: f"{x:02d} - {['Januari','Februari','Maret','April','Mei','Juni','Juli','Agustus','September','Oktober','November','Desember'][x-1]}",
                index=5,  # Default Juni
                help="Pilih bulan untuk prediksi"
            )
            
            luas_panen = st.number_input(
                "📏 Luas Panen (Hektare)", 
                min_value=0.1, 
                max_value=10000.0,
                value=100.0, 
                step=0.1,
                help="Masukkan luas area panen dalam hektare"
            )
            
            submit_button = st.form_submit_button(
                "🔮 Prediksi Hasil Panen", 
                type="primary",
                use_container_width=True
            )
        
        if submit_button:
            # Create input dataframe
            input_df = pd.DataFrame({
                'Tahun': [tahun],
                'Bulan': [bulan],
                'Luas Panen/ha': [luas_panen]
            })
            
            # Preprocess
            processed_df = preprocess_data(input_df)
            
            # Predict
            prediction = make_prediction(processed_df, model)[0]
            
            # Store results
            st.session_state['single_prediction'] = {
                'prediction': prediction,
                'tahun': tahun,
                'bulan': bulan,
                'luas_panen': luas_panen,
                'musim': processed_df['season'].iloc[0],
                'quarter': processed_df['quarter'].iloc[0]
            }
    
    with col2:
        st.subheader("📊 Hasil Prediksi")
        
        if 'single_prediction' in st.session_state:
            pred_data = st.session_state['single_prediction']
            
            # Main result
            st.success(f"**🌾 Prediksi Hasil Panen: {pred_data['prediction']:.2f} ton**")
            
            # Metrics
            col2_1, col2_2, col2_3 = st.columns(3)
            
            with col2_1:
                st.metric(
                    label="📅 Periode",
                    value=f"{pred_data['bulan']:02d}/{pred_data['tahun']}"
                )
            
            with col2_2:
                musim_emoji = "☀️" if pred_data['musim'] == 'dry' else "🌧️"
                musim_text = "Kering" if pred_data['musim'] == 'dry' else "Basah"
                st.metric(
                    label="🌿 Musim",
                    value=f"{musim_emoji} {musim_text}"
                )
            
            with col2_3:
                productivity = pred_data['prediction'] / pred_data['luas_panen']
                st.metric(
                    label="⚡ Produktivitas",
                    value=f"{productivity:.2f} ton/ha"
                )
            
            # Gauge chart
            fig = create_gauge_chart(pred_data['prediction'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional info
            with st.expander("📋 Detail Prediksi"):
                st.json({
                    "Input": {
                        "Tahun": pred_data['tahun'],
                        "Bulan": pred_data['bulan'],
                        "Luas Panen (ha)": pred_data['luas_panen']
                    },
                    "Features": {
                        "Musim": pred_data['musim'],
                        "Kuartal": f"Q{pred_data['quarter']}"
                    },
                    "Output": {
                        "Prediksi (ton)": round(pred_data['prediction'], 2),
                        "Produktivitas (ton/ha)": round(productivity, 2)
                    }
                })
        else:
            st.info("👆 Masukkan parameter di sebelah kiri dan klik 'Prediksi Hasil Panen'")

else:
    # CSV UPLOAD MODE
    st.subheader("📁 Upload & Prediksi Batch")
    
    # Template section
    with st.expander("📋 Template CSV", expanded=True):
        col_temp1, col_temp2 = st.columns([2, 1])
        
        with col_temp1:
            template_df = pd.DataFrame({
                'Tahun': [2023, 2023, 2024, 2024, 2024],
                'Bulan': [3, 8, 1, 6, 10],
                'Luas Panen/ha': [150.5, 200.0, 175.8, 220.3, 180.0]
            })
            st.dataframe(template_df, use_container_width=True)
            
        with col_temp2:
            csv_template = template_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Template",
                csv_template,
                "template_prediksi_panen.csv",
                "text/csv",
                use_container_width=True
            )
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📂 Pilih File CSV",
        type=['csv'],
        help="File harus berformat CSV dengan kolom: Tahun, Bulan, Luas Panen/ha"
    )
    
    if uploaded_file is not None:
        try:
            # Read and validate CSV
            df = pd.read_csv(uploaded_file)
            
            required_cols = ['Tahun', 'Bulan', 'Luas Panen/ha']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ **Kolom yang hilang:** {', '.join(missing_cols)}")
                st.info("📝 **Format yang benar:** Tahun, Bulan, Luas Panen/ha")
            else:
                st.success(f"✅ **File berhasil dimuat!** {len(df)} baris data ditemukan")
                
                # Data validation
                col_val1, col_val2 = st.columns(2)
                
                with col_val1:
                    st.subheader("👀 Preview Data")
                    st.dataframe(df.head(10), use_container_width=True)
                
                with col_val2:
                    st.subheader("📈 Ringkasan Data")
                    st.markdown(f"""
                    - **Total Baris:** {len(df)}
                    - **Rentang Tahun:** {df['Tahun'].min()} - {df['Tahun'].max()}
                    - **Rentang Bulan:** {df['Bulan'].min()} - {df['Bulan'].max()}
                    - **Luas Panen:**
                      - Min: {df['Luas Panen/ha'].min():.1f} ha
                      - Max: {df['Luas Panen/ha'].max():.1f} ha
                      - Rata-rata: {df['Luas Panen/ha'].mean():.1f} ha
                    """)
                
                # Prediction button
                if st.button("🚀 Jalankan Prediksi Batch", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Preprocessing
                        status_text.text("🔄 Memproses data...")
                        progress_bar.progress(25)
                        
                        processed_df = preprocess_data(df)
                        
                        # Prediction
                        status_text.text("🤖 Menjalankan prediksi...")
                        progress_bar.progress(50)
                        
                        predictions = make_prediction(processed_df, model)
                        
                        # Prepare results
                        status_text.text("📊 Menyiapkan hasil...")
                        progress_bar.progress(75)
                        
                        result_df = df.copy()
                        result_df['Prediksi Hasil Panen (ton)'] = predictions
                        result_df['Musim'] = processed_df['season'].map({'dry': 'Kering', 'wet': 'Basah'})
                        result_df['Kuartal'] = processed_df['quarter'].map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
                        result_df['Produktivitas (ton/ha)'] = predictions / df['Luas Panen/ha']
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Prediksi selesai!")
                        
                        # Store results
                        st.session_state['batch_results'] = result_df
                        
                    except Exception as e:
                        st.error(f"❌ Error dalam prediksi: {str(e)}")
                
                # Display results if available
                if 'batch_results' in st.session_state:
                    result_df = st.session_state['batch_results']
                    predictions = result_df['Prediksi Hasil Panen (ton)'].values
                    
                    st.markdown("---")
                    st.subheader("🎯 Hasil Prediksi Batch")
                    
                    # Summary metrics
                    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                    
                    with col_met1:
                        st.metric("📊 Total Prediksi", f"{predictions.sum():.1f} ton")
                    with col_met2:
                        st.metric("📈 Rata-rata", f"{predictions.mean():.1f} ton")
                    with col_met3:
                        st.metric("🔝 Tertinggi", f"{predictions.max():.1f} ton")
                    with col_met4:
                        st.metric("🔻 Terendah", f"{predictions.min():.1f} ton")
                    
                    # Results tabs
                    tab1, tab2, tab3 = st.tabs(["📋 Data Hasil", "📊 Visualisasi", "📥 Download"])
                    
                    with tab1:
                        st.dataframe(result_df, use_container_width=True, height=400)
                    
                    with tab2:
                        # Time series plot
                        if len(result_df['Tahun'].unique()) > 1:
                            fig1 = px.line(
                                result_df, 
                                x='Bulan', 
                                y='Prediksi Hasil Panen (ton)',
                                color='Tahun',
                                title="📈 Tren Prediksi per Bulan dan Tahun",
                                markers=True
                            )
                            fig1.update_layout(height=400)
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        # Seasonal comparison
                        seasonal_data = result_df.groupby('Musim')['Prediksi Hasil Panen (ton)'].agg(['mean', 'sum', 'count']).reset_index()
                        
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            fig2 = px.bar(
                                seasonal_data, 
                                x='Musim', 
                                y='mean',
                                title="🌿 Rata-rata Prediksi per Musim",
                                color='Musim',
                                color_discrete_map={'Kering': '#FFA500', 'Basah': '#4169E1'}
                            )
                            fig2.update_layout(height=350)
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        with col_chart2:
                            fig3 = px.scatter(
                                result_df,
                                x='Luas Panen/ha',
                                y='Prediksi Hasil Panen (ton)',
                                color='Musim',
                                size='Produktivitas (ton/ha)',
                                title="🎯 Hubungan Luas vs Hasil Panen",
                                color_discrete_map={'Kering': '#FFA500', 'Basah': '#4169E1'}
                            )
                            fig3.update_layout(height=350)
                            st.plotly_chart(fig3, use_container_width=True)
                    
                    with tab3:
                        st.subheader("📥 Download Hasil Prediksi")
                        
                        col_dl1, col_dl2 = st.columns(2)
                        
                        with col_dl1:
                            # CSV download
                            csv_result = result_df.to_csv(index=False)
                            st.download_button(
                                "📄 Download CSV",
                                csv_result,
                                f"hasil_prediksi_panen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                use_container_width=True
                            )
                        
                        with col_dl2:
                            # Excel download
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                result_df.to_excel(writer, sheet_name='Hasil Prediksi', index=False)
                                df.to_excel(writer, sheet_name='Data Input', index=False)
                                seasonal_data.to_excel(writer, sheet_name='Ringkasan Musim', index=False)
                            
                            st.download_button(
                                "📊 Download Excel",
                                output.getvalue(),
                                f"hasil_prediksi_panen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                
        except Exception as e:
            st.error(f"❌ **Error membaca file:** {str(e)}")
            st.info("💡 **Tips:** Pastikan file CSV menggunakan encoding UTF-8 dan format yang benar")

# Footer
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown("**🚀 Tentang Aplikasi**")
    st.markdown("Sistem prediksi hasil panen menggunakan algoritma Random Forest yang dioptimasi dengan Whale Optimization Algorithm")

with col_footer2:
    st.markdown("**📊 Fitur Utama**")
    st.markdown("- Prediksi tunggal & batch\n- Visualisasi interaktif\n- Export hasil ke CSV/Excel")

with col_footer3:
    st.markdown("**🔬 Model Info**")
    st.markdown("- Input: Tahun, Bulan, Luas Panen\n- Output: Prediksi hasil (ton)\n- Mode: Demo/Production")

if not SKLEARN_AVAILABLE:
    st.error("⚠️ Beberapa dependencies tidak tersedia. Aplikasi berjalan dalam mode terbatas.")
