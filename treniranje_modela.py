import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'student_processed.csv')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'izvestaj_grafikoni')
REZULTATI_PATH = os.path.join(CURRENT_DIR, 'rezultati.txt')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DEFINIŠEMO TAČNE ATRIBUTE (Isto kao u tvojim najnovijim izmenama)
FEAT_SA  = ['G2', 'G1', 'Medu', 'Fedu', 'Dalc', 'reason_other']
FEAT_BEZ = ['failures', 'Dalc', 'Medu', 'Fedu', 'studytime', 'Walc', 'higher', 'reason_other', 'reason_reputation', 'absences', 'goout']


def izracunaj_metrike(y_test, y_pred, naziv, log=None):
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    linija = f"  {naziv:<30} MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}"
    print(linija)
    if log:
        log.append(linija)
    return {'Model': naziv, 'MAE': round(mae,3), 'RMSE': round(rmse,3), 'R2': round(r2,3)}


def pokreni_treniranje(data_path, output_dir, rezultati_path):
    print("TRENIRANJE I PODEŠAVANJE HIPERPARAMETARA")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Nema podataka na: {data_path}\nPokreni prvo priprema_podataka.py!")

    df = pd.read_csv(data_path)
    print(f"\nUčitan dataset: {df.shape[0]} redova, {df.shape[1]} kolona")

    # ══════════════════════════════════════════════════════════════
    # 1. PAMETNO FILTRIRANJE ANOMALIJA (G3=0 i absences=0)
    # ══════════════════════════════════════════════════════════════
    anomalija = (df['G3'] == 0) & (df['absences'] == 0)
    df = df[~anomalija]
    print(f"Dataset nakon filtriranja (bez lažnih nula): {df.shape[0]} redova")

    y = df['G3']

    # ══════════════════════════════════════════════════════════════
    # 2. SELEKCIJA TAČNIH SKUPOVA (Uzimamo samo izabrane kolone!)
    # ══════════════════════════════════════════════════════════════
    skupovi = {
        'Sa G1 i G2':  df[FEAT_SA],
        'Bez G1 i G2': df[FEAT_BEZ],
    }

    # Dublja stabla i manji listovi dozvoljavaju modelu da "nauči" niske ocene
    param_tree = {
        'max_depth':         [5, 7, 10, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf':  [1, 2]
    }
    param_forest = {
        'n_estimators':      [100, 200],
        'max_depth':         [10, 15, None],
        'min_samples_split': [2, 5]
    }

    rez_sa   = []
    rez_bez  = []
    log_linije = []

    log_linije.append("REZULTATI TRENIRANJA I PODEŠAVANJA HIPERPARAMETARA")


    najbolji = {}

    for naziv_skupa, X in skupovi.items():
        print(f"\nSKUP: {naziv_skupa.upper()}")

        log_linije.append("")
        log_linije.append("=" * 56)
        log_linije.append(f"REZULTATI — {naziv_skupa.upper()}")
        log_linije.append("=" * 56)

        # Delimo na 80-20 (krosvalidacija unutar Grid-a radi ostalo, nema potrebe za val splitom ovde)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42)

        print(f"Podela: Train={len(X_train)} | Test={len(X_test)}")
        log_linije.append(f"Podela: Train={len(X_train)} | Test={len(X_test)}")

        # ── DEO A: Default Modeliri ───────────────────────────
        print("\nDEO A: Treniranje sa podrazumevanim parametrima")
        log_linije.append("")
        log_linije.append("DEO A: Default modeli")


        default_modeli = {
            'Linearna regresija': LinearRegression(),
            'Stablo odlučivanja': DecisionTreeRegressor(random_state=42, max_depth=5),
            'Slučajna šuma':      RandomForestRegressor(random_state=42, n_estimators=100),
        }

        default_predikcije = {}

        for naziv, model in default_modeli.items():
            model.fit(X_train, y_train)
            yp = np.clip(np.round(model.predict(X_test)), 0, 20)
            default_predikcije[naziv] = yp
            rez = izracunaj_metrike(y_test, yp, naziv, log_linije)
            if naziv_skupa == 'Sa G1 i G2':
                rez_sa.append(rez)
            else:
                rez_bez.append(rez)

        # ── DEO B: Tuning (Optimizujemo MAE preko neg_mean_absolute_error) ──
        print("\nPodešavanje hiperparametara — GridSearchCV")
        log_linije.append("")
        log_linije.append(" Nakon tuninga (GridSearchCV)")

        print("\n  [1/2] Stablo odlučivanja...")
        gs_tree = GridSearchCV(
            DecisionTreeRegressor(random_state=42),
            param_tree, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        gs_tree.fit(X_train, y_train)
        log_linije.append(f"  Stablo — parametri: {gs_tree.best_params_}")

        yp_tree = np.clip(np.round(gs_tree.best_estimator_.predict(X_test)), 0, 20)
        rez_tree = izracunaj_metrike(y_test, yp_tree, 'Stablo odlučivanja (tuned)', log_linije)
        if naziv_skupa == 'Sa G1 i G2':
            rez_sa.append(rez_tree)
        else:
            rez_bez.append(rez_tree)

        print("\n  [2/2] Slučajna šuma...")
        gs_forest = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_forest, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        gs_forest.fit(X_train, y_train)
        log_linije.append(f"  Šuma  — parametri: {gs_forest.best_params_}")

        yp_forest = np.clip(np.round(gs_forest.best_estimator_.predict(X_test)), 0, 20)
        rez_forest = izracunaj_metrike(y_test, yp_forest, 'Slučajna šuma (tuned)', log_linije)
        if naziv_skupa == 'Sa G1 i G2':
            rez_sa.append(rez_forest)
        else:
            rez_bez.append(rez_forest)

        svi_rez = rez_sa if naziv_skupa == 'Sa G1 i G2' else rez_bez
        najbolji[naziv_skupa] = max(svi_rez, key=lambda x: x['R2'])

        # Iscrtavanje grafikona
        naziv_safe = naziv_skupa.replace(' ', '_').replace('/', '')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Stvarne vs Predviđene vrednosti — {naziv_skupa}', fontsize=12, fontweight='bold')
        boje = ['#4A90D9', '#2ECC71', '#E85D75']
        for ax, (naziv_def, yp_def), boja in zip(axes, default_predikcije.items(), boje):
            ax.scatter(y_test, yp_def, alpha=0.5, color=boja, edgecolors='white', s=40)
            lims = [min(y_test.min(), yp_def.min())-0.5, max(y_test.max(), yp_def.max())+0.5]
            ax.plot(lims, lims, 'k--', lw=1.2, label='Idealno')
            ax.set_xlabel('Stvarna ocena G3')
            ax.set_ylabel('Predviđena ocena G3')
            ax.set_title(naziv_def)
            ax.legend(fontsize=8)
            r2_def = r2_score(y_test, yp_def)
            ax.text(0.05, 0.92, f'R²={r2_def:.3f}', transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'stvarno_vs_predvidjeno_{naziv_safe}.png'), bbox_inches='tight')
        plt.close()

        if naziv_skupa == 'Bez G1 i G2':
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle('Važnost atributa — Bez G1 i G2 (tuned modeli)', fontsize=12, fontweight='bold')
            for ax, (naziv_m, model_fit) in zip(axes, [
                ('Slučajna šuma (tuned)',      gs_forest.best_estimator_),
                ('Stablo odlučivanja (tuned)', gs_tree.best_estimator_),
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

    # Grafikoni poređenja
    df_sa  = pd.DataFrame(rez_sa[0:3]) # Uzimamo default modele za uporedni grafik
    df_bez = pd.DataFrame(rez_bez[0:3])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Poređenje modela — Sa G1/G2 vs Bez G1/G2', fontsize=13, fontweight='bold')
    for ax, metrika, naslov in zip(axes, ['MAE', 'RMSE', 'R2'], ['MAE (niže = bolje)', 'RMSE (niže = bolje)', 'R² (više = bolje)']):
        x = np.arange(len(df_sa))
        w = 0.35
        bars1 = ax.bar(x - w/2, df_sa[metrika], w, label='Sa G1/G2', color='#4A90D9', edgecolor='white')
        bars2 = ax.bar(x + w/2, df_bez[metrika], w, label='Bez G1/G2', color='#E85D75', edgecolor='white')
        for bar in list(bars1) + list(bars2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)
        ax.set_title(naslov)
        ax.set_xticks(x)
        ax.set_xticklabels(df_sa['Model'], rotation=15, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'poredjenje_modela.png'), bbox_inches='tight')
    plt.close()

    log_linije.append("")
    log_linije.append("ZAKLJUČAK")
    for skup, rez in najbolji.items():
        oznaka = " PREPORUČENI MODEL" if rez['R2'] == max(r['R2'] for r in najbolji.values()) else ""
        log_linije.append(f"Najbolji model {skup}:")
        log_linije.append(f"  Model : {rez['Model']}")
        log_linije.append(f"  MAE   = {rez['MAE']}  RMSE = {rez['RMSE']}  R² = {rez['R2']}  {oznaka}")

    with open(rezultati_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_linije))
    print(f"\nRezultati upisani u: {rezultati_path}")


if __name__ == "__main__":
    pokreni_treniranje(PROCESSED_PATH, OUTPUT_DIR, REZULTATI_PATH)