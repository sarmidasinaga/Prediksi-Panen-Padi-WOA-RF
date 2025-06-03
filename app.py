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

# Whale Optimization Algorithm (WOA) Class
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

st.title('Analisis & Prediksi Hasil Panen - WOA + Random Forest')

uploaded_file = st.file_uploader("Upload file data (CSV)...", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Contoh Data", df.head())

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
    st.write("### Korelasi Fitur (heatmap)")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)

    # --- Preprocessing ---
    X = df.drop(['Hasil Panen/ton'], axis=1)
    y = df['Hasil Panen/ton']
    if 'season' in X.columns:
        le = LabelEncoder()
        X['season_encoded'] = le.fit_transform(X['season'])
        X = X.drop(['season'], axis=1)

    # --- Split dan Scaling ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- RF Initial Feature Importance ---
    rf_basic = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_basic.fit(X_train_scaled, y_train)
    feat_imp = pd.DataFrame({'feature': X.columns, 'importance': rf_basic.feature_importances_})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    st.write("### Feature Importance (RF Initial)")
    st.dataframe(feat_imp)

    # --- Whale Optimization Algorithm (WOA) for Hyperparameter Tuning ---
    st.info("Menjalankan Whale Optimization untuk hyperparameter tuning RF... Mohon tunggu (±2 menit untuk dataset besar)")

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
        (50, 200),      # n_estimators (dibatasi agar tidak lama)
        (3, 12),        # max_depth
        (2, 10),        # min_samples_split
        (1, 5),         # min_samples_leaf
        (0.3, 1.0)      # max_features
    ]
    woa = WhaleOptimization(
        n_whales=8,         # agar cepat
        n_iterations=12,    # agar cepat, boleh tambah jika mau hasil lebih optimal
        bounds=bounds,
        objective_function=rf_objective_function
    )
    best_params, best_fitness, fitness_history = woa.optimize()
    st.success(f"Best parameters: n_estimators={int(best_params[0])}, max_depth={int(best_params[1])}, "
               f"min_samples_split={int(best_params[2])}, min_samples_leaf={int(best_params[3])}, "
               f"max_features={best_params[4]:.2f} (MSE: {best_fitness:.2f})")

    # --- Tampilkan grafik konvergensi fitness ---
    st.write("### Konvergensi Whale Optimization (MSE terbaik per iterasi)")
    fig2, ax2 = plt.subplots()
    ax2.plot(fitness_history, marker='o')
    ax2.set_xlabel("Iterasi")
    ax2.set_ylabel("Best MSE")
    st.pyplot(fig2)

    # --- Training final model RF dengan best params ---
    final_rf = RandomForestRegressor(
        n_estimators=int(best_params[0]),
        max_depth=int(best_params[1]),
        min_samples_split=int(best_params[2]),
        min_samples_leaf=int(best_params[3]),
        max_features=best_params[4],
        random_state=42, n_jobs=-1
    )
    final_rf.fit(X_train_scaled, y_train)

    # --- Predict dan evaluasi ---
    y_train_pred = final_rf.predict(X_train_scaled)
    y_test_pred = final_rf.predict(X_test_scaled)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    st.write("### Performance Metrics")
    st.write("#### Training Set", train_metrics)
    st.write("#### Test Set", test_metrics)

    # --- Visualisasi Prediksi vs Aktual ---
    st.write("### Grafik Prediksi vs Aktual (Test set)")
    df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
    fig3, ax3 = plt.subplots()
    df_pred.plot(kind='bar', ax=ax3)
    plt.xlabel("Index")
    plt.ylabel("Hasil Panen/ton")
    plt.legend(['Actual', 'Predicted'])
    st.pyplot(fig3)
