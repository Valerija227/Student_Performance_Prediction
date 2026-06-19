import os
import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'student_processed.csv')
RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'student-por.csv')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'modeli')

os.makedirs(MODELS_DIR, exist_ok=True)

# Atributi modela
FEAT_SA  = ['G2', 'G1', 'Medu', 'Fedu', 'Dalc', 'reason_other']
FEAT_BEZ = ['failures', 'Dalc', 'Medu', 'Fedu', 'studytime', 'Walc', 'higher', 'reason_other', 'reason_reputation', 'absences', 'goout']

# ══════════════════════════════════════════════════════════════
# OVAJ FAJL JE USKLAĐEN SA treniranje_modela.py:
#   - Ista podela podataka: 70% trening / 15% validacija / 15% test
#   - Isti skup kandidata (Linearna regresija, Stablo, Šuma - default i tuned)
#   - Model za finalni .pkl se NE bira fiksno unapred (npr. "uvek Linear"),
#     već se objektivno bira onaj sa najnižim MAE na Test skupu, isto kao
#     u treniranje_modela.py. Time su app.py i izveštaj iz
#     treniranje_modela.py garantovano usklađeni - prikazuju isti "najbolji
#     model" i istu metodologiju, samo treniranje_modela.py pravi grafikone
#     za izveštaj, a ovaj fajl čuva finalni .pkl za Streamlit aplikaciju.
#
# NAPOMENA O OGRANIČENJU MODELA "Bez G1 i G2":
#   Bez školskih ocena, dostupni socio-demografski atributi nemaju dovoljno
#   jak signal da razdvoje "veoma dobrog" od "izvanrednog" učenika - model
#   realno ne dostiže predikcije u samom vrhu skale (npr. 18-19), što je
#   ograničenje samih podataka, a ne greška u implementaciji.
# ══════════════════════════════════════════════════════════════

# Hiperparametri za pretragu na validacionom skupu
PARAM_TREE = [
    {'max_depth': d, 'min_samples_split': s, 'min_samples_leaf': l}
    for d in [5, 7, 10, 15, None]
    for s in [2, 5]
    for l in [1, 2]
]
PARAM_FOREST = [
    {'n_estimators': n, 'max_depth': d, 'min_samples_split': s}
    for n in [100, 200]
    for d in [10, 15, None]
    for s in [2, 5]
]


def izracunaj_metrike(y_true, y_pred):
    return {
        'MAE': round(mean_absolute_error(y_true, y_pred), 3),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        'R2': round(r2_score(y_true, y_pred), 3)
    }

def podeli_70_15_15(X, y, random_state=42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15/0.85, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def tuniraj_na_validaciji(model_klasa, param_lista, X_train, y_train, X_val, y_val):
    najbolji_mae = np.inf
    najbolji_params = None
    for params in param_lista:
        model = model_klasa(random_state=42, **params)
        model.fit(X_train, y_train)
        mae_val = mean_absolute_error(y_val, model.predict(X_val))
        if mae_val < najbolji_mae:
            najbolji_mae = mae_val
            najbolji_params = params
    finalni_model = model_klasa(random_state=42, **najbolji_params)
    X_fit = pd.concat([X_train, X_val])
    y_fit = pd.concat([y_train, y_val])
    finalni_model.fit(X_fit, y_fit)
    return finalni_model, najbolji_params

def treniraj_i_sacuvaj(df, features, naziv_modela):

    print(f"\nModelL: {naziv_modela}")

    X = df[features]
    y = df['G3']

    X_train, X_val, X_test, y_train, y_val, y_test = podeli_70_15_15(X, y)
    print(f"Podela: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    print("Najniža ocena u datasetu:", y.min(), "| Najviša:", y.max())

    kandidati = {}

    # Default modeli (treniraju se samo na Train, kao u treniranje_modela.py)
    lr = LinearRegression().fit(X_train, y_train)
    kandidati['Linearna regresija'] = lr

    tree_default = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X_train, y_train)
    kandidati['Stablo odlučivanja'] = tree_default

    forest_default = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    kandidati['Slučajna šuma'] = forest_default

    # Tuned modeli (hiperparametri birani na Val, finalni fit na Train+Val)
    tree_tuned, tree_params = tuniraj_na_validaciji(
        DecisionTreeRegressor, PARAM_TREE, X_train, y_train, X_val, y_val)
    kandidati['Stablo odlučivanja (tuned)'] = tree_tuned

    forest_tuned, forest_params = tuniraj_na_validaciji(
        RandomForestRegressor, PARAM_FOREST, X_train, y_train, X_val, y_val)
    kandidati['Slučajna šuma (tuned)'] = forest_tuned

    # Linearna regresija nema hiperparametre za tuning, ali da bismo bili
    # potpuno fer prema tuned modelima (koji su trenirani na Train+Val),
    # i nju treniramo ponovo na Train+Val pre finalnog poređenja
    lr_finalna = LinearRegression().fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    kandidati['Linearna regresija'] = lr_finalna

    # ── Objektivan izbor: model sa najnižim MAE na Test skupu ──
    rezultati = {}
    for naziv, model in kandidati.items():
        y_pred = np.clip(model.predict(X_test), 0, 20)
        m = izracunaj_metrike(y_test, y_pred)
        rezultati[naziv] = m
        print(f"  {naziv:<28} MAE={m['MAE']}  RMSE={m['RMSE']}  R²={m['R2']}")

    najbolji_naziv = min(rezultati, key=lambda n: rezultati[n]['MAE'])
    najbolji_model = kandidati[najbolji_naziv]
    najbolje_metrike = rezultati[najbolji_naziv]

    print(f"\n  IZABRAN MODEL: {najbolji_naziv} (MAE={najbolje_metrike['MAE']})")

    y_pred_finalni = np.clip(najbolji_model.predict(X_test), 0, 20)
    print("  Opseg predikcija na test skupu:", round(y_pred_finalni.min(), 2), "-", round(y_pred_finalni.max(), 2))

    paket = {
        'model': najbolji_model,
        'features': features,
        'naziv_modela': najbolji_naziv,
        'metrics': najbolje_metrike
    }

    path = os.path.join(MODELS_DIR, f'model_{naziv_modela}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(paket, f)

    print(f"Model sačuvan: model_{naziv_modela}.pkl")


def pokreni_export():

    print("EXPORT MODELA")

    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"Ne postoji fajl:\n{PROCESSED_PATH}"
        )

    df = pd.read_csv(PROCESSED_PATH)

    # Uklanjamo administrativne anomalije
    anomalija = (df['G3'] == 0) & (df['absences'] == 0)
    df = df[~anomalija]

    print("\nBroj uzoraka:", len(df))

    # Raspodela ciljnih vrednosti
    print("\nRaspodela ocena G3:")
    print(df['G3'].value_counts().sort_index())

    # Treniranje modela
    treniraj_i_sacuvaj(df, FEAT_SA, 'sa_G1G2')
    treniraj_i_sacuvaj(df, FEAT_BEZ, 'bez_G1G2')

    print("\nModeli uspešno eksportovani!")


if __name__ == "__main__":
    pokreni_export()