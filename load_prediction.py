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

CUSTOM_CSS = """
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background:
            radial-gradient(1200px 500px at 10% -10%, rgba(59,130,246,0.16), transparent 60%),
            radial-gradient(900px 500px at 100% 0%, rgba(16,185,129,0.10), transparent 55%),
            #0F1720;
    }

    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px;}

    /* Header banner */
    .app-header {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.75rem;
        background: linear-gradient(135deg, #16233A 0%, #0F1720 100%);
        border: 1px solid rgba(59,130,246,0.25);
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
        color: #E6EDF3;
        letter-spacing: -0.01em;
    }
    .app-header p {
        margin: 0.35rem 0 0 0;
        color: #93A4B8;
        font-size: 0.95rem;
    }
    .app-header .badge {
        display: inline-block;
        margin-top: 0.75rem;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        background: rgba(59,130,246,0.15);
        border: 1px solid rgba(59,130,246,0.35);
        color: #7EB2FF;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Section cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #182233;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    }

    h3 {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #C7D3E0 !important;
        border-left: 3px solid #3B82F6;
        padding-left: 0.6rem;
        margin-bottom: 1rem !important;
    }

    .stButton > button, .stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.4rem;
        border: none;
        box-shadow: 0 4px 12px rgba(59,130,246,0.35);
    }

    /* Result card */
    .result-card {
        padding: 1.5rem 1.75rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(59,130,246,0.14), rgba(16,185,129,0.08));
        border: 1px solid rgba(59,130,246,0.3);
        margin-top: 0.5rem;
    }
    .result-card .label {
        color: #93A4B8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .result-card .value {
        color: #E6EDF3;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0.2rem 0 0.4rem 0;
    }
    .result-card .range {
        color: #93A4B8;
        font-size: 0.85rem;
    }

    [data-testid="stMetricValue"] {color: #E6EDF3;}
    hr {border-color: rgba(255,255,255,0.08);}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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

st.markdown(
    """
    <div class="app-header">
        <h1>⚡ Prediksi LOAD</h1>
        <p>Estimasi beban (LOAD) dari parameter operasi menggunakan model deep learning regresi.</p>
        <span class="badge">dl_regresi_load_es.keras · 18 fitur input</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("input_form"):
    with st.container(border=True):
        st.subheader("Parameter operasi")
        c1, c2, c3, c4 = st.columns(4)
        values = {}
        values["CF"] = number_input(c1, "CF", X_ref, 0.1, "%.4f")
        values["SFC"] = number_input(c2, "SFC", X_ref, 0.01, "%.6f")
        values["O2 PCT"] = number_input(c3, "O2 PCT", X_ref, 0.1, "%.4f")
        values["FT"] = number_input(c4, "FT", X_ref, 1.0, "%.4f")

    with st.container(border=True):
        st.subheader("Kondisi (0 = tidak, 1 = ya)")
        cols = st.columns(len(BINARY_FEATURES))
        for col, name in zip(cols, BINARY_FEATURES):
            values[name] = col.selectbox(
                name, [0, 1], index=int(X_ref[name].median()), key=f"bin_{name}"
            )

    with st.container(border=True):
        st.subheader("Komposisi")
        c1, c2, c3, c4, c5 = st.columns(5)
        values["NK LRCP"] = number_input(c1, "NK LRCP", X_ref, 1.0, "%.0f")
        values["NK LRC"] = number_input(c2, "NK LRC", X_ref, 1.0, "%.0f")
        values["% LRCP"] = number_input(c3, "% LRCP", X_ref, 1.0, "%.0f")
        values["% LRC"] = number_input(c4, "% LRC", X_ref, 1.0, "%.0f")
        values["BIO"] = c5.selectbox(
            "BIO", BIO_OPTIONS, index=BIO_OPTIONS.index(int(X_ref["BIO"].median()))
        )

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Prediksi", type="primary", use_container_width=True)

if submitted:
    # Urutan kolom harus sama dengan saat training
    new_data = pd.DataFrame([values], columns=feature_order)
    scaled = scaler.transform(new_data)
    prediction = float(model.predict(scaled, verbose=0)[0][0])
    out_of_range = not (y_ref.min() <= prediction <= y_ref.max())

    st.markdown(
        f"""
        <div class="result-card">
            <div class="label">Prediksi LOAD</div>
            <div class="value">{prediction:,.3f}</div>
            <div class="range">
                Rentang LOAD pada data latih: {y_ref.min():,.2f} - {y_ref.max():,.2f}
                (rata-rata {y_ref.mean():,.2f})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if out_of_range:
        st.warning("Hasil prediksi berada di luar rentang LOAD pada data latih.")

    with st.expander("Data input yang dipakai"):
        st.dataframe(new_data, hide_index=True)
