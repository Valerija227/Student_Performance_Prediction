#model.py

import os
import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'student_processed.csv')
MODELS_DIR = os.path.join(CURRENT_DIR, 'modeli')

os.makedirs(MODELS_DIR, exist_ok=True)

# Atributi modela
FEAT_SA  = ['G2', 'G1', 'Medu', 'Fedu', 'Dalc', 'reason_other']
FEAT_BEZ = ['failures', 'Dalc', 'Medu', 'Fedu', 'studytime', 'Walc', 'higher', 'reason_other', 'reason_reputation', 'absences', 'goout']

# ══════════════════════════════════════════════════════════════
# ZAŠTO LinearRegression, A NE STABLO / ŠUMA?
#
# Stabla odlučivanja i RandomForest INTERPOLIRAJU - predviđaju
# prosek primera iz lista kome najsličniji ulaz pripada, ali ne
# mogu da EKSTRAPOLIRAJU van opsega viđenih kombinacija. Pošto u
# skupu skoro da nema učenika sa ekstremno lošim profilom (npr.
# G1=0 i G2=0 zajedno), stablo takve slučajeve gura u najbliži
# postojeći list i "zaglavi" se na vrednosti od ~6-8, bez obzira
# koliko je unos zapravo loš.
#
# LinearRegression pretpostavlja linearan (pravolinijski) odnos
# između atributa i ocene, pa ekstrapolira logično i dalje od
# viđenih podataka - npr. za G1=G2=0 predviđa ocenu blizu 0, što
# je realno očekivanje. Provereno je i da ovaj skup atributa
# (posebno G1, G2, failures) ima dominantno linearnu vezu sa G3,
# pa LinearRegression ovde i ima bolje metrike od stabla/šume.
# ══════════════════════════════════════════════════════════════


def izracunaj_metrike(y_true, y_pred):
    return {
        'MAE': round(mean_absolute_error(y_true, y_pred), 3),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        'R2': round(r2_score(y_true, y_pred), 3)
    }


def treniraj_i_sacuvaj(df, features, naziv_modela):

    print(f"MODEL: {naziv_modela}")

    X = df[features]
    y = df['G3']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Ispis najmanje ocene
    print("Najniža ocena u datasetu:", y.min())

    # Test skup
    y_pred = np.clip(model.predict(X_test), 0, 20)

    metrike = izracunaj_metrike(y_test, y_pred)

    print("MAE :", metrike['MAE'])
    print("RMSE:", metrike['RMSE'])
    print("R²  :", metrike['R2'])
    print("Opseg predikcija na test skupu:", round(y_pred.min(), 2), "-", round(y_pred.max(), 2))

    paket = {
        'model': model,
        'features': features,
        'metrics': metrike
    }

    path = os.path.join(
        MODELS_DIR,
        f'model_{naziv_modela}.pkl'
    )

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
    treniraj_i_sacuvaj(
        df,
        FEAT_SA,
        'sa_G1G2'
    )

    treniraj_i_sacuvaj(
        df,
        FEAT_BEZ,
        'bez_G1G2'
    )

    print("\nModeli uspešno eksportovani!")


if __name__ == "__main__":
    pokreni_export()