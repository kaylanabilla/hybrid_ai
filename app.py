import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Hybrid AI Productivity Predictor",
    layout="centered"
)

st.title("📊 Hybrid AI Productivity Predictor")
st.write("Model: **Naive Bayes + ANN (Stacking)**")

# ============================
# LOAD DATA
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("Morning_Routine_Productivity_Dataset.csv")

    # ==== FIX 1: GANTI DATE -> Tanggal (AMAN) ====
    if "DATE" in df.columns:
        df = df.rename(columns={"DATE": "Tanggal"})

    # ==== FIX 2: PASTIKAN Tanggal ADA ====
    if "Tanggal" not in df.columns:
        st.error("❌ Kolom DATE / Tanggal tidak ditemukan di dataset")
        st.stop()

    # ==== FIX 3: KONVERSI TANGGAL ====
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    df["Tanggal"] = df["Tanggal"].fillna(df["Tanggal"].mode()[0])
    df["Tanggal"] = df["Tanggal"].map(pd.Timestamp.toordinal)

    # ==== FIX 4: ENCODER DISIMPAN ====
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Target
    df["Produktif"] = (df["Productivity_Score (1-10)"] >= 7).astype(int)

    return df, encoders

df, encoders = load_data()

# ============================
# FEATURES & TARGET
# ============================
X = df.drop(columns=["Productivity_Score (1-10)", "Produktif"])
y = df["Produktif"]

# ==== FIX 5: SIMPAN URUTAN KOLOM ====
feature_columns = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ============================
# TRAIN MODELS
# ============================
@st.cache_resource
def train_models():
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    proba_nb = nb.predict_proba(X_test)[:, 1]

    ann = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    ann.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    ann.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        verbose=0
    )

    proba_ann = ann.predict(X_test).flatten()

    X_meta = np.column_stack((proba_nb, proba_ann))
    meta = LogisticRegression(max_iter=1000)
    meta.fit(X_meta, y_test)

    f1 = f1_score(y_test, meta.predict(X_meta))
    return nb, ann, meta, f1

nb_model, ann_model, meta_model, f1_model = train_models()

st.success(f"✅ Model trained | F1 Score: {f1_model:.4f}")

# ============================
# USER INPUT
# ============================
st.subheader("🧠 Input Data Pengguna")

user_data = {}

for col in feature_columns:
    if col == "Tanggal":
        tgl = st.date_input("Tanggal", value=date.today())
        user_data[col] = pd.Timestamp(tgl).toordinal()

    elif col in encoders:
        pilihan = encoders[col].classes_.tolist()
        pilih = st.selectbox(col, pilihan)
        user_data[col] = encoders[col].transform([pilih])[0]

    else:
        user_data[col] = st.number_input(
            col,
            value=float(df[col].mean())
        )

# ============================
# PREDICTION
# ============================
if st.button("🔍 Predict"):
    # ==== FIX 6: URUTAN KOLOM DIJAMIN SAMA ====
    user_df = pd.DataFrame([[user_data[col] for col in feature_columns]],
                           columns=feature_columns)

    user_scaled = scaler.transform(user_df)

    nb_prob = nb_model.predict_proba(user_scaled)[:, 1]
    ann_prob = ann_model.predict(user_scaled).flatten()

    meta_input = np.column_stack((nb_prob, ann_prob))
    result = meta_model.predict(meta_input)[0]

    if result == 1:
        st.success("🎯 Prediction: **PRODUKTIF**")
    else:
        st.error("⚠️ Prediction: **TIDAK PRODUKTIF**")
