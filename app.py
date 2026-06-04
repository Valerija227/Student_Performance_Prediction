import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ─────────────────────────────────────────────────────────────
# UČITAVANJE MODELA
# ─────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(CURRENT_DIR, 'modeli')

@st.cache_resource
def ucitaj_modele():
    modeli = {}
    for naziv in ['sa_G1G2', 'bez_G1G2']:
        path = os.path.join(MODELS_DIR, f'model_{naziv}.pkl')
        with open(path, 'rb') as f:
            modeli[naziv] = pickle.load(f)
    return modeli

# UI
st.set_page_config(page_title="Predikcija ocene učenika", page_icon="🎓", layout="wide")
st.title("🎓 Predikcija završne ocene učenika (G3)")
st.markdown("Unesite podatke o učeniku i dobijte predviđenu završnu ocenu.")

modeli = ucitaj_modele()

# ── Izbor režima ─────────────────────────────────────────────
st.sidebar.header("⚙️ Podešavanja")
rezim = st.sidebar.radio(
    "Izaberite režim predikcije:",
    ["Sa ocenama G1 i G2", "Bez ocena G1 i G2"],
    help="Sa ocenama je tačniji model (R²=0.87). Bez ocena koristi samo demografske faktore (R²=0.42)."
)
kljuc = 'sa_G1G2' if rezim == "Sa ocenama G1 i G2" else 'bez_G1G2'
paket = modeli[kljuc]

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Performanse modela")
m = paket['metrics']
st.sidebar.metric("R²",   m['R2'])
st.sidebar.metric("MAE",  m['MAE'])
st.sidebar.metric("RMSE", m['RMSE'])

# ── Forma — isti izgled za oba režima ────────────────────────
st.markdown("---")
st.subheader("📝 Podaci o učeniku")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Ocene**")
    G1 = st.slider("Ocena u prvom periodu (G1)", 0, 20, 10)
    G2 = st.slider("Ocena u drugom periodu (G2)", 0, 20, 10)
    failures = st.selectbox(
        "Broj prethodnih neuspeha",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"{x} {'neuspeha' if x != 1 else 'neuspeh'}"
    )

with col2:
    st.markdown("**Obrazovanje i učenje**")
    studytime = st.selectbox(
        "Nedeljno vreme učenja",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1:"< 2 sata", 2:"2–5 sati", 3:"5–10 sati", 4:"> 10 sati"}[x]
    )
    Medu = st.selectbox(
        "Obrazovanje majke",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: {0:"Bez obrazovanja", 1:"Osnovna škola",
                                2:"5–9. razred", 3:"Srednja škola", 4:"Visoko obrazovanje"}[x]
    )
    Fedu = st.selectbox(
        "Obrazovanje oca",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: {0:"Bez obrazovanja", 1:"Osnovna škola",
                                2:"5–9. razred", 3:"Srednja škola", 4:"Visoko obrazovanje"}[x]
    )
    higher = st.selectbox(
        "Želi visoko obrazovanje?",
        options=[1, 0],
        format_func=lambda x: "Da" if x == 1 else "Ne"
    )

with col3:
    st.markdown("**Socijalni faktori**")
    Dalc = st.slider("Konzumacija alkohola radnim danom (1=malo, 5=puno)", 1, 5, 1)
    Walc = st.slider("Konzumacija alkohola vikendom (1=malo, 5=puno)", 1, 5, 1)
    reason_other = st.selectbox(
        "Razlog upisa — drugi razlog?",
        options=[0, 1],
        format_func=lambda x: "Da" if x == 1 else "Ne"
    )
    reason_reputation = st.selectbox(
        "Reputacija škole kao razlog upisa?",
        options=[0, 1],
        format_func=lambda x: "Da" if x == 1 else "Ne"
    )

# ── Predikcija ───────────────────────────────────────────────
st.markdown("---")

if st.button("🔮 Predvidi ocenu", type="primary", use_container_width=True):

    # Svi mogući atributi
    svi_atributi = {
        'G1': G1, 'G2': G2,
        'failures': failures, 'studytime': studytime,
        'Medu': Medu, 'Fedu': Fedu,
        'higher': higher,
        'Dalc': Dalc, 'Walc': Walc,
        'reason_other': reason_other,
        'reason_reputation': reason_reputation,
    }

    # Uzimamo samo atribute koje model koristi
    ulaz = pd.DataFrame([{k: svi_atributi[k] for k in paket['features']}])
    ulaz = ulaz[paket['features']]

    predikcija = paket['model'].predict(ulaz)[0]
    predikcija = int(np.clip(round(predikcija), 0, 20))

    col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
    with col_r2:
        if predikcija >= 15:
            boja = "🟢"
            komentar = "Odličan učenik!"
        elif predikcija >= 10:
            boja = "🟡"
            komentar = "Prosečan učenik."
        else:
            boja = "🔴"
            komentar = "Učenik ima poteškoća."

        st.markdown(f"""
        <div style='text-align:center; padding:30px;
                    background-color:#f0f2f6; border-radius:15px;'>
            <h1 style='font-size:80px; margin:0'>{boja}</h1>
            <h2>Predviđena ocena: <strong>{predikcija}/20</strong></h2>
            <p style='font-size:18px'>{komentar}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Student Performance Prediction — SAUSAU projekat")