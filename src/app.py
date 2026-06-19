#app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# UČITAVANJE MODELA

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

st.set_page_config(page_title="Predikcija ocene učenika", page_icon="🎓", layout="wide")
st.title("🎓 Predikcija završne ocene učenika (G3)")
st.markdown("Unesite podatke o učeniku i dobijte predviđenu završnu ocenu.")

modeli = ucitaj_modele()

# ── Izbor režima ─────────────────────────────────────────────
st.sidebar.header("⚙️ Podešavanja")
rezim = st.sidebar.radio(
    "Izaberite režim predikcije:",
    ["Sa ocenama G1 i G2", "Bez ocena G1 i G2"],
    help="Sa ocenama koristi školske rezultate. Bez ocena koristi isključivo socio-demografske faktore."
)
kljuc = 'sa_G1G2' if rezim == "Sa ocenama G1 i G2" else 'bez_G1G2'
paket = modeli[kljuc]

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Performanse modela")
m = paket['metrics']
st.sidebar.metric("R²",   m['R2'])
st.sidebar.metric("MAE",  m['MAE'])
st.sidebar.metric("RMSE", m['RMSE'])

st.markdown("---")
st.subheader("📝 Podaci o učeniku")

# Rečnik u koji pakujemo sve korisničke unose
svi_atributi = {}

# ══════════════════════════════════════════════════════════════
# FORMA — SA G1 I G2 (FEAT_SA: G2, G1, Medu, Fedu, Dalc, reason_other)
# ══════════════════════════════════════════════════════════════
if kljuc == 'sa_G1G2':
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Školski uspeh**")
        svi_atributi['G1'] = st.slider("Ocena u prvom periodu (G1)", 0, 20, 10)
        svi_atributi['G2'] = st.slider("Ocena u drugom periodu (G2)", 0, 20, 10)

    with col2:
        st.markdown("**Porodica i ciljevi**")
        svi_atributi['Medu'] = st.selectbox(
            "Obrazovanje majke",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: {0:"Bez obrazovanja", 1:"Osnovna škola",
                                   2:"5–9. razred", 3:"Srednja škola",
                                   4:"Visoko obrazovanje"}[x],
            key="medu_sa"
        )
        svi_atributi['Fedu'] = st.selectbox(
            "Obrazovanje oca",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: {0:"Bez obrazovanja", 1:"Osnovna škola",
                                   2:"5–9. razred", 3:"Srednja škola",
                                   4:"Visoko obrazovanje"}[x],
            key="fedu_sa"
        )
        razlog_sa = st.selectbox(
            "Razlog izbora škole",
            options=["course", "home", "reputation", "other"],
            format_func=lambda x: {"course":"Smer/predmeti", "home":"Blizina kuće",
                                   "reputation":"Reputacija škole", "other":"Drugo"}[x],
            key="reason_sa"
        )
        svi_atributi['reason_other'] = 1 if razlog_sa == "other" else 0

    with col3:
        st.markdown("**Životne navike**")
        svi_atributi['Dalc'] = st.slider("Alkohol radnim danom (1=malo, 5=puno)", 1, 5, 1, key="dalc_sa")

# ══════════════════════════════════════════════════════════════
# FORMA — BEZ G1 I G2
# (FEAT_BEZ: failures, Dalc, Medu, Fedu, studytime, Walc, higher,
#            reason_other, reason_reputation, absences, goout)
# ══════════════════════════════════════════════════════════════
else:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Akademsko okruženje**")

        svi_atributi['failures'] = st.selectbox(
            "Broj prethodnih padova / neuspeha",
            options=[0, 1, 2, 3],
            format_func=lambda x: f"{x} neuspeha" if x != 1 else "1 neuspeh"
        )
        svi_atributi['studytime'] = st.selectbox(
            "Nedeljno vreme učenja",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1:"< 2 sata", 2:"2–5 sati",
                                   3:"5–10 sati", 4:"> 10 sati"}[x]
        )
        svi_atributi['higher'] = st.selectbox(
            "Želi visoko obrazovanje?",
            options=[1, 0],
            format_func=lambda x: "Da" if x == 1 else "Ne",
            key="higher_bez"
        )

    with col2:
        st.markdown("**Porodični status**")
        svi_atributi['Medu'] = st.selectbox(
            "Obrazovanje majke",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: {0:"Bez obrazovanja", 1:"Osnovna škola",
                                   2:"5–9. razred", 3:"Srednja škola",
                                   4:"Visoko obrazovanje"}[x],
            key="medu_bez"
        )
        svi_atributi['Fedu'] = st.selectbox(
            "Obrazovanje oca",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: {0:"Bez obrazovanja", 1:"Osnovna škola",
                                   2:"5–9. razred", 3:"Srednja škola",
                                   4:"Visoko obrazovanje"}[x],
            key="fedu_bez"
        )
        razlog_bez = st.selectbox(
            "Razlog izbora škole",
            options=["course", "home", "reputation", "other"],
            format_func=lambda x: {"course":"Smer/predmeti", "home":"Blizina kuće",
                                   "reputation":"Reputacija škole", "other":"Drugo"}[x],
            key="reason_bez"
        )
        svi_atributi['reason_other'] = 1 if razlog_bez == "other" else 0
        svi_atributi['reason_reputation'] = 1 if razlog_bez == "reputation" else 0

    with col3:
        st.markdown("**Slobodno vreme i navike**")
        svi_atributi['absences'] = st.slider("Broj izostanaka", 0, 93, 0, key="abs_bez")
        svi_atributi['Dalc'] = st.slider("Alkohol radnim danom (1=malo, 5=puno)", 1, 5, 1, key="dalc_bez")
        svi_atributi['Walc'] = st.slider("Konzumacija alkohola vikendom (1=malo, 5=puno)", 1, 5, 1, key="walc_bez")
        svi_atributi['goout'] = st.slider("Učestalost izlazaka sa prijateljima (1=malo, 5=puno)", 1, 5, 3, key="goout_bez")

# ── Predikcija ───────────────────────────────────────────────
st.markdown("---")

if st.button("🔮 Predvidi ocenu", type="primary", use_container_width=True):

    # Dinamički kreiramo DataFrame prateći tačan redosled kolona koji zahteva učitani model
    ulaz = pd.DataFrame([{k: svi_atributi[k] for k in paket['features']}])
    ulaz = ulaz[paket['features']]

    predikcija = paket['model'].predict(ulaz)[0]
    predikcija = float(np.clip(predikcija, 0, 20))
    predikcija_prikaz = int(round(predikcija))

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
            komentar = "Učenik ima poteškoća u savladavanju gradiva."

        st.markdown(f"""
        <div style='text-align:center; padding:30px;
                    background-color:#f0f2f6; border-radius:15px;'>
            <h1 style='font-size:80px; margin:0'>{boja}</h1>
            <h2>Predviđena ocena: <strong>{predikcija_prikaz}/20</strong></h2>
            <p style='font-size:18px'>{komentar}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Student Performance Prediction — Napredni ML Model")