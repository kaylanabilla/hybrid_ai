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
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Hybrid AI Productivity Predictor",
    page_icon="🚀",
    layout="centered"
)

# ==================================================
# CUSTOM CSS (UI KEREN)
# ==================================================
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}
h1, h2, h3 {
    color: #38bdf8;
}
.card {
    background: linear-gradient(135deg, #1e293b, #020617);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 0 25px rgba(56,189,248,0.15);
    margin-bottom: 20px;
}
.result-good {
    background: linear-gradient(135deg, #16a34a, #15803d);
    padding: 20px;
    border-radius: 18px;
    color: white;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
.result-bad {
    background: linear-gradient(135deg, #dc2626, #991b1b);
    padding: 20px;
    border-radius: 18px;
    color: white;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
.footer {
    text-align: center;
    opacity: 0.7;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="card">
<h1>🚀 Hybrid AI Productivity Predictor</h1>
<p>Model AI cerdas berbasis <b>Naive Bayes + Artificial Neural Network (Stacking)</b>
untuk memprediksi tingkat produktivitas harian.</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA
# ==================================================
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

# ==================================================
# TRAIN MODEL
# ==================================================
@st.cache_resource
def train_models():
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    ann = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    ann.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    proba_nb = nb.predict_proba(X_test)[:, 1]
    proba_ann = ann.predict(X_test).flatten()

    meta = LogisticRegression(max_iter=1000)
    meta.fit(np.column_stack((proba_nb, proba_ann)), y_test)

    f1 = f1_score(y_test, meta.predict(np.column_stack((proba_nb, proba_ann))))

    return nb, ann, meta, f1

nb_model, ann_model, meta_model, f1_model = train_models()

st.markdown(f"""
<div class="card">
<h3>📈 Model Performance</h3>
<p><b>F1 Score:</b> {f1_model:.4f}</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# INPUT SECTION
# ==================================================
st.markdown("""
<div class="card">
<h3>🧠 Input Data Pagi Hari</h3>
<p>Masukkan parameter rutinitas pagimu</p>
</div>
""", unsafe_allow_html=True)

user_input = []
for col in X.columns:
    val = st.slider(
        col,
        float(X[col].min()),
        float(X[col].max()),
        float(X[col].mean())
    )
    user_input.append(val)

# ==================================================
# PREDICTION
# ==================================================
if st.button("🔮 PREDIKSI PRODUKTIVITAS", use_container_width=True):
    user_array = np.array(user_input).reshape(1, -1)
    user_scaled = scaler.transform(user_array)

    p_nb = nb_model.predict_proba(user_scaled)[:, 1]
    p_ann = ann_model.predict(user_scaled).flatten()

    meta_input = np.column_stack((p_nb, p_ann))
    result = meta_model.predict(meta_input)[0]
    confidence = meta_model.predict_proba(meta_input)[0][result] * 100

    st.markdown("### 📊 Hasil Prediksi")

    st.progress(int(confidence))

    if result == 1:
        st.markdown(f"""
        <div class="result-good">
        ✅ PRODUKTIF<br>
        Confidence: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-bad">
        ❌ TIDAK PRODUKTIF<br>
        Confidence: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<div class="footer">
<p>© 2025 Hybrid AI System | Naive Bayes + ANN + Stacking</p>
</div>
""", unsafe_allow_html=True)

