from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "dl_regresi_load_es.keras"
DATA_PATH = BASE_DIR / "Pred_Load_Labuan.csv"
TARGET = "LOAD"

# Kolom bernilai 0/1 (ya/tidak) dan kolom numerik kontinu/diskrit
BINARY_FEATURES = ["BA", "OBP", "GEL", "SGP", "SPE", "TBC", "EEI", "SSN", "ARM"]
BIO_OPTIONS = [1, 2, 3]

st.set_page_config(page_title="Prediksi LOAD", page_icon="⚡", layout="wide")


@st.cache_resource(show_spinner="Memuat model...")
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner="Menyiapkan data referensi...")
def get_reference():
    """Baca data latih dan bangun ulang scaler seperti di notebook.

    Scaler tidak disimpan bersama model, jadi di-fit ulang pada training set dengan
    split yang sama seperti saat training (test_size=0.2, random_state=0).
    """
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    X = df.drop(columns=[TARGET])
    X_train, _, _, _ = train_test_split(
        X, df[TARGET], test_size=0.2, random_state=0
    )
    scaler = MinMaxScaler().fit(X_train)
    return X, df[TARGET], scaler


def number_input(col, label, stats, step, fmt):
    lo, hi = float(stats[label].min()), float(stats[label].max())
    return col.number_input(
        label,
        min_value=lo,
        max_value=hi,
        value=float(stats[label].median()),
        step=step,
        format=fmt,
        help=f"Rentang data latih: {lo:g} - {hi:g}",
    )


try:
    model = get_model()
    X_ref, y_ref, scaler = get_reference()
except FileNotFoundError as e:
    st.error(f"File tidak ditemukan: {e.filename}. Pastikan berada di folder yang sama dengan load_prediction.py.")
    st.stop()

feature_order = list(X_ref.columns)

st.title("⚡ Prediksi LOAD")
st.caption("Model deep learning regresi (`dl_regresi_load_es.keras`) dengan 18 fitur input.")

with st.form("input_form"):
    st.subheader("Parameter operasi")
    c1, c2, c3, c4 = st.columns(4)
    values = {}
    values["CF"] = number_input(c1, "CF", X_ref, 0.1, "%.4f")
    values["SFC"] = number_input(c2, "SFC", X_ref, 0.01, "%.6f")
    values["O2 PCT"] = number_input(c3, "O2 PCT", X_ref, 0.1, "%.4f")
    values["FT"] = number_input(c4, "FT", X_ref, 1.0, "%.4f")

    st.subheader("Kondisi (0 = tidak, 1 = ya)")
    cols = st.columns(len(BINARY_FEATURES))
    for col, name in zip(cols, BINARY_FEATURES):
        values[name] = col.selectbox(
            name, [0, 1], index=int(X_ref[name].median()), key=f"bin_{name}"
        )

    st.subheader("Komposisi")
    c1, c2, c3, c4, c5 = st.columns(5)
    values["NK LRCP"] = number_input(c1, "NK LRCP", X_ref, 1.0, "%.0f")
    values["NK LRC"] = number_input(c2, "NK LRC", X_ref, 1.0, "%.0f")
    values["% LRCP"] = number_input(c3, "% LRCP", X_ref, 1.0, "%.0f")
    values["% LRC"] = number_input(c4, "% LRC", X_ref, 1.0, "%.0f")
    values["BIO"] = c5.selectbox(
        "BIO", BIO_OPTIONS, index=BIO_OPTIONS.index(int(X_ref["BIO"].median()))
    )

    submitted = st.form_submit_button("Prediksi", type="primary")

if submitted:
    # Urutan kolom harus sama dengan saat training
    new_data = pd.DataFrame([values], columns=feature_order)
    scaled = scaler.transform(new_data)
    prediction = float(model.predict(scaled, verbose=0)[0][0])

    st.divider()
    st.metric("Prediksi LOAD", f"{prediction:,.3f}")
    st.caption(
        f"Rentang LOAD pada data latih: {y_ref.min():,.2f} - {y_ref.max():,.2f} "
        f"(rata-rata {y_ref.mean():,.2f})"
    )
    if not (y_ref.min() <= prediction <= y_ref.max()):
        st.warning("Hasil prediksi berada di luar rentang LOAD pada data latih.")

    with st.expander("Data input yang dipakai"):
        st.dataframe(new_data, hide_index=True)
