import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Hybrid AI Productivity Predictor", layout="centered")

st.title("📊 Hybrid AI Productivity Predictor")
st.write("Model: **Naive Bayes + Artificial Neural Network (Stacking)**")

MODEL_PATH = "model_hybrid.pkl"
DATA_PATH = "Morning_Routine_Productivity_Dataset.csv"

# =========================
# TRAIN MODEL (JALAN SEKALI)
# =========================
def train_and_save_model():
    df = pd.read_csv(DATA_PATH)

    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    df["Produktif"] = (df["Productivity_Score (1-10)"] >= 7).astype(int)

    X = df.drop(columns=["Productivity_Score (1-10)", "Produktif"])
    y = df["Produktif"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, _, y_train, _ = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # ANN
    ann = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    ann.compile(optimizer="adam", loss="binary_crossentropy")
    ann.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Meta learner (BENAR: pakai TRAIN)
    proba_nb = nb.predict_proba(X_train)[:, 1]
    proba_ann = ann.predict(X_train).flatten()
    X_meta = np.column_stack((proba_nb, proba_ann))

    meta = LogisticRegression(max_iter=1000)
    meta.fit(X_meta, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "nb": nb,
            "ann": ann,
            "meta": meta,
            "columns": X.columns.tolist()
        }, f)

# =========================
# LOAD MODEL (TANPA TRAIN)
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

scaler = model["scaler"]
nb_model = model["nb"]
ann_model = model["ann"]
meta_model = model["meta"]
columns = model["columns"]

st.success("✅ Model siap digunakan (tanpa training ulang)")

# =========================
# INPUT USER
# =========================
st.subheader("🧠 Input Data")

cols = st.columns(2)
user_input = []

for i, col in enumerate(columns):
    with cols[i % 2]:
        val = st.number_input(col, value=0.0)
        user_input.append(val)

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Predict"):
    user_array = np.array(user_input).reshape(1, -1)
    user_scaled = scaler.transform(user_array)

    nb_p = nb_model.predict_proba(user_scaled)[:, 1]
    ann_p = ann_model.predict(user_scaled).flatten()

    meta_input = np.column_stack((nb_p, ann_p))
    confidence = meta_model.predict_proba(meta_input)[0][1] * 100

    st.progress(int(confidence))

    if confidence >= 80:
        st.success(f"🔥 **SANGAT PRODUKTIF** ({confidence:.2f}%)")
    elif confidence >= 60:
        st.success(f"✅ **PRODUKTIF** ({confidence:.2f}%)")
    elif confidence >= 40:
        st.warning(f"⚖️ **CUKUP PRODUKTIF** ({confidence:.2f}%)")
    else:
        st.error(f"❌ **TIDAK PRODUKTIF** ({confidence:.2f}%)")

