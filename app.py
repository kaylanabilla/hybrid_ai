import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# ==================================================
# Page Config
# ==================================================
st.set_page_config(
    page_title="Hybrid AI Productivity Predictor",
    layout="centered"
)

st.title("📊 Hybrid AI Productivity Predictor")
st.write("Model: **Naive Bayes + ANN (Stacking)**")

# ==================================================
# Load Dataset
# ==================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Morning_Routine_Productivity_Dataset.csv")

    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        if col not in ["date", "notes"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    df["Produktif"] = (df["Productivity_Score (1-10)"] >= 7).astype(int)
    return df, encoders

df, encoders = load_data()

# ==================================================
# Feature Selection
# ==================================================
ignore_cols = ["date", "notes"]

X = df.drop(
    columns=["Productivity_Score (1-10)", "Produktif"] + ignore_cols,
    errors="ignore"
)
y = df["Produktif"]

# ==================================================
# Scaling
# ==================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==================================================
# Train Models
# ==================================================
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
    ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    ann.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    proba_ann = ann.predict(X_test).flatten()

    X_meta = np.column_stack((proba_nb, proba_ann))
    meta = LogisticRegression(max_iter=1000)
    meta.fit(X_meta, y_test)

    f1 = f1_score(y_test, meta.predict(X_meta))
    return nb, ann, meta, f1

nb_model, ann_model, meta_model, f1_model = train_models()
st.success(f"✅ Model trained | F1 Score: {f1_model:.4f}")

# ==================================================
# FORM INPUT (DATE & NOTES DIKETIK MANUAL)
# ==================================================
st.subheader("📝 Input Data Harian")

with st.form("daily_form"):
    col1, col2 = st.columns(2)
    with col1:
        input_date = st.text_input(
            "Tanggal",
            placeholder="Contoh: 2025-01-10"
        )
    with col2:
        input_notes = st.text_area(
            "Catatan / Notes",
            placeholder="Tulis catatan harian di sini"
        )

    st.markdown("### 🧠 Data untuk Prediksi")

    user_data = {}
    for col in X.columns:
        if col in encoders:
            pilihan = encoders[col].classes_
            selected = st.selectbox(col, pilihan)
            user_data[col] = encoders[col].transform([selected])[0]
        else:
            user_data[col] = st.number_input(
                col,
                value=float(df[col].mean())
            )

    submitted = st.form_submit_button("🔍 Predict Productivity")

# ==================================================
# Prediction
# ==================================================
if submitted:
    user_df = pd.DataFrame([user_data])
    user_scaled = scaler.transform(user_df)

    nb_prob = nb_model.predict_proba(user_scaled)[:, 1]
    ann_prob = ann_model.predict(user_scaled).flatten()

    meta_input = np.column_stack((nb_prob, ann_prob))
    result = meta_model.predict(meta_input)[0]

    st.markdown("## 📊 Hasil Prediksi")
    st.write("📅 Tanggal:", input_date)
    st.write("📝 Catatan:", input_notes)

    if result == 1:
        st.success("🎯 HASIL: **PRODUKTIF**")
    else:
        st.error("⚠️ HASIL: **TIDAK PRODUKTIF**")
