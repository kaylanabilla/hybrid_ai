
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

st.set_page_config(page_title="Hybrid AI Productivity Predictor", layout="centered")

st.title("📊 Hybrid AI Productivity Predictor")
st.write("Model: **Naive Bayes + Artificial Neural Network (Stacking)**")

# ============================
# Load Dataset
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
# Train Models
# ============================
@st.cache_resource
def train_models():
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    proba_nb = nb.predict_proba(X_test)[:, 1]

    # ANN
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

    # Meta Learner
    X_meta = np.column_stack((proba_nb, proba_ann))
    meta = LogisticRegression(max_iter=1000)
    meta.fit(X_meta, y_test)

    f1 = f1_score(y_test, meta.predict(X_meta))

    return nb, ann, meta, f1

nb_model, ann_model, meta_model, f1_score_model = train_models()

st.success(f"✅ Model trained successfully | F1 Score: {f1_score_model:.4f}")

# ============================
# User Input
# ============================
st.subheader("🧠 Predict Productivity")

user_input = []
for col in X.columns:
    val = st.number_input(f"{col}", value=float(X[col].mean()))
    user_input.append(val)

if st.button("🔍 Predict"):
    user_array = np.array(user_input).reshape(1, -1)
    user_scaled = scaler.transform(user_array)

    nb_p = nb_model.predict_proba(user_scaled)[:, 1]
    ann_p = ann_model.predict(user_scaled).flatten()

    meta_input = np.column_stack((nb_p, ann_p))
    result = meta_model.predict(meta_input)[0]
    confidence = meta_model.predict_proba(meta_input)[0][result] * 100

    st.markdown("## 📊 Hasil Prediksi Produktivitas")

    # =========================
    # TINGKAT PRODUKTIVITAS
    # =========================
    if confidence >= 80:
        st.success(f"🔥 **SANGAT PRODUKTIF** ({confidence:.2f}%)")
        st.progress(100)
    elif confidence >= 60:
        st.success(f"✅ **PRODUKTIF** ({confidence:.2f}%)")
        st.progress(int(confidence))
    elif confidence >= 40:
        st.warning(f"⚖️ **CUKUP PRODUKTIF** ({confidence:.2f}%)")
        st.progress(int(confidence))
    else:
        st.error(f"❌ **TIDAK PRODUKTIF** ({confidence:.2f}%)")
        st.progress(int(confidence))

