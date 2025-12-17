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

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Hybrid AI Productivity Predictor", layout="centered")

# =========================
# CSS (TAMPILAN SAJA)
# =========================
st.markdown("""
<style>
.title-box {
    background: linear-gradient(135deg,#1e293b,#020617);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
.result-ok {
    background: linear-gradient(135deg,#16a34a,#15803d);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 22px;
    font-weight: bold;
}
.result-bad {
    background: linear-gradient(135deg,#dc2626,#991b1b);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 22px;
    font-weight: bold;
}
.small-text {
    opacity: 0.85;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="title-box">
<h1>📊 Hybrid AI Productivity Predictor</h1>
<p>Model: <b>Naive Bayes + Artificial Neural Network (Stacking)</b></p>
</div>
""", unsafe_allow_html=True)

# ============================
# LOAD DATASET (ASLI)
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("Morning_Routine_Productivity_Dataset.csv")
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    df["Produktif"] = (df["Productivity_Score (1-10)"] >= 7).astype(int)
    return df

df = load_data()

X = df.drop(columns=["Productivity_Score (1-10)", "Produktif"])
y = df["Produktif"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ============================
# TRAIN MODEL (ASLI)
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
    ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    ann.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    proba_ann = ann.predict(X_test).flatten()

    X_meta = np.column_stack((proba_nb, proba_ann))
    meta = LogisticRegression(max_iter=1000)
    meta.fit(X_meta, y_test)

    f1 = f1_score(y_test, meta.predict(X_meta))
    return nb, ann, meta, f1

nb_model, ann_model, meta_model, f1_score_model = train_models()

st.info(f"📈 **F1 Score Model:** {f1_score_model:.4f}")

# ============================
# INPUT USER (KIRI – KANAN)
# ============================
st.subheader("🧠 Input Data")

cols = st.columns(2)
user_input = []

for i, col in enumerate(X.columns):
    with cols[i % 2]:
        val = st.number_input(
            label=col,
            value=float(X[col].mean())
        )
        user_input.append(val)

# ============================
# PREDIKSI
# ============================
if st.button("🔍 Predict"):
    user_array = np.array(user_input).reshape(1, -1)
    user_scaled = scaler.transform(user_array)

    nb_p = nb_model.predict_proba(user_scaled)[:, 1]
    ann_p = ann_model.predict(user_scaled).flatten()

    meta_input = np.column_stack((nb_p, ann_p))
    result = meta_model.predict(meta_input)[0]
    confidence = meta_model.predict_proba(meta_input)[0][result] * 100

    st.progress(int(confidence))

    if result == 1:
        st.markdown(f"""
        <div class="result-ok">
        ✅ PRODUKTIF<br>
        <span class="small-text">Confidence: {confidence:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-bad">
        ❌ TIDAK PRODUKTIF<br>
        <span class="small-text">Confidence: {confidence:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
