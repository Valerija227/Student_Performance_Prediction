#app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# UČITAVANJE MODELA
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'modeli')


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
st.sidebar.caption(f"Model: {paket['naziv_modela']}")

st.markdown("---")
st.subheader("📝 Podaci o učeniku")

svi_atributi = {}

def unesi_zajednicka_polja(svi_atributi, key_suffix):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Akademsko okruženje**")
        skola_izbor = st.selectbox(
            "Škola",
            options=["GP", "MS"],
            format_func=lambda x: {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"}[x],
            key=f"school_{key_suffix}"
        )
        # prepare_data.py enkodira binarne kolone alfabetski (astype('category').cat.codes),
        # za 'school' to znači GP -> 0, MS -> 1. Model očekuje broj, ne string.
        svi_atributi['school'] = 0 if skola_izbor == "GP" else 1
        svi_atributi['failures'] = st.selectbox(
            "Broj prethodnih padova / neuspeha",
            options=[0, 1, 2, 3],
            format_func=lambda x: f"{x} neuspeha" if x != 1 else "1 neuspeh",
            key=f"failures_{key_suffix}"
        )
        svi_atributi['studytime'] = st.selectbox(
            "Nedeljno vreme učenja",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "< 2 sata", 2: "2–5 sati",
                                    3: "5–10 sati", 4: "> 10 sati"}[x],
            key=f"studytime_{key_suffix}"
        )
        svi_atributi['higher'] = st.selectbox(
            "Želi visoko obrazovanje?",
            options=[1, 0],
            format_func=lambda x: "Da" if x == 1 else "Ne",
            key=f"higher_{key_suffix}"
        )

    with col2:
        st.markdown("**Porodični status**")
        svi_atributi['Medu'] = st.selectbox(
            "Obrazovanje majke",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: {0: "Bez obrazovanja", 1: "Osnovna škola",
                                    2: "5–9. razred", 3: "Srednja škola",
                                    4: "Visoko obrazovanje"}[x],
            key=f"medu_{key_suffix}"
        )
        mjob = st.selectbox(
            "Zanimanje majke",
            options=["teacher", "health", "services", "at_home", "other"],
            format_func=lambda x: {"teacher": "Nastavnica/profesorka", "health": "Zdravstvo",
                                    "services": "Usluge", "at_home": "Domaćica",
                                    "other": "Drugo"}[x],
            key=f"mjob_{key_suffix}"
        )
        svi_atributi['Mjob_teacher'] = 1 if mjob == "teacher" else 0

    with col3:
        st.markdown("**Slobodno vreme i navike**")
        svi_atributi['absences'] = st.slider("Broj izostanaka", 0, 93, 0, key=f"abs_{key_suffix}")
        svi_atributi['Dalc'] = st.slider("Alkohol radnim danom (1=malo, 5=puno)", 1, 5, 1, key=f"dalc_{key_suffix}")
        svi_atributi['Walc'] = st.slider("Konzumacija alkohola vikendom (1=malo, 5=puno)", 1, 5, 1, key=f"walc_{key_suffix}")

    return svi_atributi


if kljuc == 'sa_G1G2':
    st.markdown("**Školski uspeh**")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        svi_atributi['G1'] = st.slider("Ocena u prvom periodu (G1)", 0, 20, 10, key="g1_sa")
    with col_g2:
        svi_atributi['G2'] = st.slider("Ocena u drugom periodu (G2)", 0, 20, 10, key="g2_sa")
    st.markdown("---")
    svi_atributi = unesi_zajednicka_polja(svi_atributi, key_suffix="sa")
else:
    svi_atributi = unesi_zajednicka_polja(svi_atributi, key_suffix="bez")

# ── Predikcija ───────────────────────────────────────────────
st.markdown("---")

if st.button("🔮 Predvidi ocenu", type="primary", use_container_width=True):

    nedostaju = [k for k in paket['features'] if k not in svi_atributi]
    if nedostaju:
        st.error(
            f"Forma ne prikuplja sledeće atribute koje model očekuje: {nedostaju}. "
            "Proveri da li su FEAT_SA/FEAT_BEZ liste u treniranje_modela.py / "
            "export_modela.py usklađene sa formom u app.py."
        )
    else:
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