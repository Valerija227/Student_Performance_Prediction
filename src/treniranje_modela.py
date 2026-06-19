import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'student_processed.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'izvestaj_grafikoni')
REZULTATI_PATH = os.path.join(PROJECT_ROOT, 'rezultati.txt')

os.makedirs(OUTPUT_DIR, exist_ok=True)
# DEFINIŠEMO TAČNE ATRIBUTE
FEAT_SA  = ['G2', 'G1', 'Medu', 'Fedu', 'Dalc', 'reason_other']
FEAT_BEZ = ['failures', 'Dalc', 'Medu', 'Fedu', 'studytime', 'Walc', 'higher', 'reason_other', 'reason_reputation', 'absences', 'goout']

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

def formatiraj_metrike(naziv, m):
    return f"  {naziv:<30} MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R²={m['R2']:.3f}"

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
        pred_val = model.predict(X_val)
        mae_val = mean_absolute_error(y_val, pred_val)
        if mae_val < najbolji_mae:
            najbolji_mae = mae_val
            najbolji_params = params
    finalni_model = model_klasa(random_state=42, **najbolji_params)
    X_fit = pd.concat([X_train, X_val])
    y_fit = pd.concat([y_train, y_val])
    finalni_model.fit(X_fit, y_fit)
    return finalni_model, najbolji_params, najbolji_mae


def pokreni_treniranje(data_path, output_dir, rezultati_path):
    print("TRENIRANJE I PODEŠAVANJE HIPERPARAMETARA")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Nema podataka na: {data_path}\nPokreni prvo priprema_podataka.py!")

    df = pd.read_csv(data_path)
    print(f"\nUčitan dataset: {df.shape[0]} redova, {df.shape[1]} kolona")

    # Uklanjamo administrativne anomalije (G3=0 uz absences=0 - odustajanje, ne nizak uspeh)
    anomalija = (df['G3'] == 0) & (df['absences'] == 0)
    df = df[~anomalija]
    print(f"Dataset nakon filtriranja (bez lažnih nula): {df.shape[0]} redova")

    y = df['G3']

    skupovi = {
        'Sa G1 i G2':  df[FEAT_SA],
        'Bez G1 i G2': df[FEAT_BEZ],
    }

    log_linije = []
    log_linije.append("REZULTATI TRENIRANJA I PODEŠAVANJA HIPERPARAMETARA")
    log_linije.append("Podela podataka: 70% trening / 15% validacija / 15% test")

    svi_rezultati = {}       # {naziv_skupa: {naziv_modela: metrike na test skupu}}
    default_predikcije_po_skupu = {}
    najbolji_modeli_po_skupu = {}  # za grafik vaznosti atributa

    for naziv_skupa, X in skupovi.items():
        print(f"\nSKUP: {naziv_skupa.upper()}")

        log_linije.append(f"REZULTATI — {naziv_skupa.upper()}")

        X_train, X_val, X_test, y_train, y_val, y_test = podeli_70_15_15(X, y)

        print(f"Podela: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
        log_linije.append(f"Podela: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

        # ── DEO A: Default modeli (treniraju se na Train, evaluiraju na Test) ──
        print("\nDEO A: Modeli sa podrazumevanim parametrima")
        log_linije.append("")
        log_linije.append("DEO A: Default modeli (trenirani na Train, evaluirani na Test)")

        default_modeli = {
            'Linearna regresija': LinearRegression(),
            'Stablo odlučivanja': DecisionTreeRegressor(random_state=42, max_depth=5),
            'Slučajna šuma':      RandomForestRegressor(random_state=42, n_estimators=100),
        }

        rezultati_skupa = {}
        default_predikcije = {}

        for naziv, model in default_modeli.items():
            model.fit(X_train, y_train)
            y_pred = np.clip(model.predict(X_test), 0, 20)
            default_predikcije[naziv] = y_pred
            m = izracunaj_metrike(y_test, y_pred)
            rezultati_skupa[naziv] = m
            linija = formatiraj_metrike(naziv, m)
            print(linija)
            log_linije.append(linija)

        # ── DEO B: Tuning hiperparametara na Validation skupu ──
        print("\nDEO B: Podešavanje hiperparametara (na Validation skupu)")
        log_linije.append("")
        log_linije.append("DEO B: Nakon tuninga (izbor hiperparametara na Validation skupu)")

        print("  [1/2] Stablo odlučivanja...")
        tree_model, tree_params, tree_val_mae = tuniraj_na_validaciji(
            DecisionTreeRegressor, PARAM_TREE, X_train, y_train, X_val, y_val)
        log_linije.append(f"  Stablo — parametri: {tree_params}  (MAE na Val={tree_val_mae:.3f})")

        y_pred_tree = np.clip(tree_model.predict(X_test), 0, 20)
        m_tree = izracunaj_metrike(y_test, y_pred_tree)
        rezultati_skupa['Stablo odlučivanja (tuned)'] = m_tree
        linija = formatiraj_metrike('Stablo odlučivanja (tuned)', m_tree)
        print(linija)
        log_linije.append(linija)

        print("  [2/2] Slučajna šuma...")
        forest_model, forest_params, forest_val_mae = tuniraj_na_validaciji(
            RandomForestRegressor, PARAM_FOREST, X_train, y_train, X_val, y_val)
        log_linije.append(f"  Šuma  — parametri: {forest_params}  (MAE na Val={forest_val_mae:.3f})")

        y_pred_forest = np.clip(forest_model.predict(X_test), 0, 20)
        m_forest = izracunaj_metrike(y_test, y_pred_forest)
        rezultati_skupa['Slučajna šuma (tuned)'] = m_forest
        linija = formatiraj_metrike('Slučajna šuma (tuned)', m_forest)
        print(linija)
        log_linije.append(linija)

        svi_rezultati[naziv_skupa] = rezultati_skupa
        default_predikcije_po_skupu[naziv_skupa] = (y_test, default_predikcije)
        najbolji_modeli_po_skupu[naziv_skupa] = {
            'Slučajna šuma (tuned)': forest_model,
            'Stablo odlučivanja (tuned)': tree_model,
        }

        # Grafikon: Stvarno vs Predviđeno (default modeli, na Test skupu)
        naziv_safe = naziv_skupa.replace(' ', '_').replace('/', '')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Stvarne vs Predviđene vrednosti — {naziv_skupa} (Test skup)', fontsize=12, fontweight='bold')
        boje = ['#4A90D9', '#2ECC71', '#E85D75']
        for ax, (naziv_def, yp_def), boja in zip(axes, default_predikcije.items(), boje):
            ax.scatter(y_test, yp_def, alpha=0.5, color=boja, edgecolors='white', s=40)
            lims = [min(y_test.min(), yp_def.min())-0.5, max(y_test.max(), yp_def.max())+0.5]
            ax.plot(lims, lims, 'k--', lw=1.2, label='Idealno')
            ax.set_xlabel('Stvarna ocena G3')
            ax.set_ylabel('Predviđena ocena G3')
            ax.set_title(naziv_def)
            ax.legend(fontsize=8)
            r2_def = rezultati_skupa[naziv_def]['R2'] if naziv_def in rezultati_skupa else r2_score(y_test, yp_def)
            ax.text(0.05, 0.92, f'R²={r2_def:.3f}', transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'stvarno_vs_predvidjeno_{naziv_safe}.png'), bbox_inches='tight')
        plt.close()

        if naziv_skupa == 'Bez G1 i G2':
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle('Važnost atributa — Bez G1 i G2 (tuned modeli)', fontsize=12, fontweight='bold')
            for ax, (naziv_m, model_fit) in zip(axes, [
                ('Slučajna šuma (tuned)',      forest_model),
                ('Stablo odlučivanja (tuned)', tree_model),
            ]):
                imp = model_fit.feature_importances_
                df_imp = pd.DataFrame({'Atribut': X.columns, 'Važnost': imp})
                df_imp = df_imp.sort_values('Važnost', ascending=True).tail(15)
                boje_bar = ['#4A90D9' if v > 0.05 else '#AEC6E8' for v in df_imp['Važnost']]
                ax.barh(df_imp['Atribut'], df_imp['Važnost'], color=boje_bar, edgecolor='white')
                ax.set_title(naziv_m)
                ax.set_xlabel('Važnost (Feature Importance)')
                ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'vaznost_atributa_bez_G1_G2.png'), bbox_inches='tight')
            plt.close()

    # Grafikon poređenja metrika (svi modeli, oba skupa)
    nazivi_modela = ['Linearna regresija', 'Stablo odlučivanja', 'Slučajna šuma',
                      'Stablo odlučivanja (tuned)', 'Slučajna šuma (tuned)']
    df_sa  = pd.DataFrame([svi_rezultati['Sa G1 i G2'][n] for n in nazivi_modela], index=nazivi_modela)
    df_bez = pd.DataFrame([svi_rezultati['Bez G1 i G2'][n] for n in nazivi_modela], index=nazivi_modela)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle('Poređenje modela — Sa G1/G2 vs Bez G1/G2 (Test skup)', fontsize=13, fontweight='bold')
    for ax, metrika, naslov in zip(axes, ['MAE', 'RMSE', 'R2'], ['MAE (niže = bolje)', 'RMSE (niže = bolje)', 'R² (više = bolje)']):
        x = np.arange(len(nazivi_modela))
        w = 0.35
        bars1 = ax.bar(x - w/2, df_sa[metrika], w, label='Sa G1/G2', color='#4A90D9', edgecolor='white')
        bars2 = ax.bar(x + w/2, df_bez[metrika], w, label='Bez G1/G2', color='#E85D75', edgecolor='white')
        for bar in list(bars1) + list(bars2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)
        ax.set_title(naslov)
        ax.set_xticks(x)
        ax.set_xticklabels(nazivi_modela, rotation=20, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'poredjenje_modela.png'), bbox_inches='tight')
    plt.close()

    # ── ZAKLJUČAK: objektivno najbolji model po MAE na Test skupu, bez forsiranja ──
    log_linije.append("")
    log_linije.append("ZAKLJUČAK")
    for naziv_skupa, rezultati_skupa in svi_rezultati.items():
        najbolji_naziv = min(rezultati_skupa, key=lambda n: rezultati_skupa[n]['MAE'])
        najbolja_metrika = rezultati_skupa[najbolji_naziv]
        log_linije.append(f"\nSkup: {naziv_skupa}")
        for naziv_m, m in rezultati_skupa.items():
            oznaka = "  <-- najbolji (najniži MAE na Test skupu)" if naziv_m == najbolji_naziv else ""
            log_linije.append(f"  {naziv_m:<30} MAE={m['MAE']}  RMSE={m['RMSE']}  R²={m['R2']}{oznaka}")

    with open(rezultati_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_linije))

if __name__ == "__main__":
    pokreni_treniranje(PROCESSED_PATH, OUTPUT_DIR, REZULTATI_PATH)