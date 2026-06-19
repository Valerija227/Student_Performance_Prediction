import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'student_processed.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'izvestaj_grafikoni')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def izracunaj_metrike(y_test, y_pred, naziv):
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"  {naziv:<35} MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    return {'Model': naziv, 'MAE': round(mae,3), 'RMSE': round(rmse,3), 'R2': round(r2,3)}


def analiziraj_skup(X, y, naziv_skupa, output_dir):
    print(f"\nSKUP: {naziv_skupa.upper()}")
    print(f"Broj atributa pre selekcije: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)
    print(f"Podela: Train={len(X_train)} | Test={len(X_test)}")

    # ── Metoda 1: Feature Importance ─────────────────────────
    print("\nFeature Importance (Random Forest)...")
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    fi_top10 = importances.head(10).index.tolist()

    # ── Metoda 2: SelectKBest ─────────────────────────────────
    print("SelectKBest (F-statistika)...")
    skb = SelectKBest(score_func=f_regression, k=10)
    skb.fit(X_train, y_train)
    skb_top10 = X.columns[skb.get_support()].tolist()
    skb_scores = pd.Series(skb.scores_, index=X.columns).sort_values(ascending=False)

    # ── Metoda 3: RFE ─────────────────────────────────────────
    print("RFE - Recursive Feature Elimination...")
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=10)
    rfe.fit(X_train, y_train)
    rfe_top10 = X.columns[rfe.support_].tolist()

    # ── Presek metoda ────────────────────────────────────────
    sve = fi_top10 + skb_top10 + rfe_top10
    selektovani = [k for k, v in Counter(sve).items() if v >= 2]

    # Poređenje
    print("\nSa SVIM atributima:")
    for naziv, model in {
        'Linearna regresija': LinearRegression(),
        'Stablo odlučivanja': DecisionTreeRegressor(random_state=42, max_depth=5),
        'Slučajna šuma':      RandomForestRegressor(random_state=42, n_estimators=100),
    }.items():
        model.fit(X_train, y_train)
        yp = np.clip(np.round(model.predict(X_test)), 0, 20)
        izracunaj_metrike(y_test, yp, naziv)

    print(f"\nSamo sa SELEKTOVANIM ({len(selektovani)}):")
    for naziv, model in {
        'Linearna regresija': LinearRegression(),
        'Stablo odlučivanja': DecisionTreeRegressor(random_state=42, max_depth=5),
        'Slučajna šuma':      RandomForestRegressor(random_state=42, n_estimators=100),
    }.items():
        model.fit(X_train[selektovani], y_train)
        yp = np.clip(np.round(model.predict(X_test[selektovani])), 0, 20)
        izracunaj_metrike(y_test, yp, naziv)

    # Iscrtavanje grafikona
    naziv_safe = naziv_skupa.replace(' ', '_').replace('/', '')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Odabir atributa — 3 metode ({naziv_skupa})', fontsize=13, fontweight='bold')

    top15 = importances.head(15).sort_values()
    boje = ['#4A90D9' if v > 0.05 else '#AEC6E8' for v in top15.values]
    axes[0].barh(top15.index, top15.values, color=boje, edgecolor='white')
    axes[0].set_title('Metoda 1: Feature Importance\n(Random Forest)')
    axes[0].grid(axis='x', alpha=0.3)

    top15_skb = skb_scores.head(15).sort_values()
    boje2 = ['#2ECC71' if k in skb_top10 else '#A8E6CF' for k in top15_skb.index]
    axes[1].barh(top15_skb.index, top15_skb.values, color=boje2, edgecolor='white')
    axes[1].set_title('Metoda 2: SelectKBest\n(F-statistika)')
    axes[1].grid(axis='x', alpha=0.3)

    rfe_ranking = pd.Series(rfe.ranking_, index=X.columns).sort_values().head(15)
    boje3 = ['#E85D75' if v == 1 else '#F4A7B2' for v in rfe_ranking.values]
    axes[2].barh(rfe_ranking.index, rfe_ranking.values, color=boje3, edgecolor='white')
    axes[2].set_title('Metoda 3: RFE\n(rang 1 = izabran)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'odabir_atributa_metode_{naziv_safe}.png'), bbox_inches='tight')
    plt.close()


def pokreni_odabir_atributa(data_path, output_dir):
    print("\nODABIR NAJZNAČAJNIJIH ATRIBUTA")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Nema podataka na: {data_path}\nPokreni prvo priprema_podataka.py!")

    df = pd.read_csv(data_path)
    
    # Filtriranje lažnih nula pre analize atributa
    anomalija = (df['G3'] == 0) & (df['absences'] == 0)
    df = df[~anomalija]
    print(f"Učitan dataset (očišćen): {df.shape[0]} redova")

    y = df['G3']

    # Skup 1: Sa G1 i G2
    analiziraj_skup(df.drop(columns=['G3']), y, 'Sa G1 i G2', output_dir)

    # Skup 2: Bez G1 i G2
    analiziraj_skup(df.drop(columns=['G3', 'G1', 'G2']), y, 'Bez G1 i G2', output_dir)


if __name__ == "__main__":
    pokreni_odabir_atributa(PROCESSED_PATH, OUTPUT_DIR)