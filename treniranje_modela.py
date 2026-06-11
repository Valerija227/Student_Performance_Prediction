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
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Računanje metrika
def izracunaj_metrike(y_test, y_pred, naziv):
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"  {naziv:<30} MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    return {'Model': naziv, 'MAE': round(mae,3), 'RMSE': round(rmse,3), 'R2': round(r2,3)}

def pokreni_treniranje(data_path, output_dir):
    print("TRENIRANJE I PODEŠAVANJE HIPERPARAMETARA")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Nema podataka na: {data_path}\nPokreni prvo priprema_podataka.py!")

    df = pd.read_csv(data_path)
    print(f"\nUčitan dataset: {df.shape[0]} redova, {df.shape[1]} kolona")

    # Ciljna promenljiva
    # G3 je ocena koju predviđamo — ona je naš "tačan odgovor"
    y = df['G3']

    # Dva skupa atributa
    # Skup 1: koristimo SVE atribute uključujući G1 i G2
    # Skup 2: koristimo SAMO demografske/socijalne faktore (bez G1, G2)
    skupovi = {
        'Sa G1 i G2':  df.drop(columns=['G3']),
        'Bez G1 i G2': df.drop(columns=['G3', 'G1', 'G2']),
    }

    # Hiperparametri za pretragu
    # Ovo su vrednosti koje GridSearch će probati za svaki model
    param_tree = {
        'max_depth':         [3, 5, 7, 10, None],  # dubina stabla
        'min_samples_split': [2, 5, 10],            # min uzoraka za grananje
        'min_samples_leaf':  [1, 2, 4]              # min uzoraka u listu
    }
    param_forest = {
        'n_estimators':      [50, 100, 200],        # broj stabala u šumi
        'max_depth':         [5, 10, None],          # dubina svakog stabla
        'min_samples_split': [2, 5, 10]             # min uzoraka za grananje
    }

    rez_sa  = []
    rez_bez = []

    for naziv_skupa, X in skupovi.items():
        print(f"\nSKUP: {naziv_skupa.upper()}")

        # Podela 70 / 15 / 15
        # Korak 1: odvajamo 30% kao privremeni skup
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42)
        # Korak 2: taj 30% delimo napola → 15% val, 15% test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42)

        print(f"Podela: Train={len(X_train)} | Validation={len(X_val)} | Test={len(X_test)}")

        # DEO A — TRENIRANJE SA DEFAULT PARAMETRIMA
        print("\nDEO A: Treniranje sa podrazumevanim parametrima")

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
            rez = izracunaj_metrike(y_test, yp, naziv)
            if naziv_skupa == 'Sa G1 i G2':
                rez_sa.append(rez)
            else:
                rez_bez.append(rez)

        # DEO B — PODEŠAVANJE HIPERPARAMETARA
        print("\nDEO B: Podešavanje hiperparametara — GridSearchCV")

        print("\n  [1/2] Stablo odlučivanja — pretražujem 45 kombinacija...")
        gs_tree = GridSearchCV(
            DecisionTreeRegressor(random_state=42),
            param_tree, cv=5, scoring='r2', n_jobs=-1)
        gs_tree.fit(X_train, y_train)
        print(f"  Najbolji parametri : {gs_tree.best_params_}")
        print(f"  CV R² na validation: {gs_tree.best_score_:.3f}")
        yp_tree = np.clip(np.round(gs_tree.best_estimator_.predict(X_test)), 0, 20)
        rez = izracunaj_metrike(y_test, yp_tree, 'Stablo odlučivanja (tuned)')
        if naziv_skupa == 'Sa G1 i G2':
            rez_sa.append(rez)
        else:
            rez_bez.append(rez)

        print("\n  [2/2] Slučajna šuma — pretražujem 27 kombinacija...")
        gs_forest = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_forest, cv=5, scoring='r2', n_jobs=-1)
        gs_forest.fit(X_train, y_train)
        print(f"  Najbolji parametri : {gs_forest.best_params_}")
        print(f"  CV R² na validation: {gs_forest.best_score_:.3f}")
        yp_forest = np.clip(np.round(gs_forest.best_estimator_.predict(X_test)), 0, 20)
        rez = izracunaj_metrike(y_test, yp_forest, 'Slučajna šuma (tuned)')
        if naziv_skupa == 'Sa G1 i G2':
            rez_sa.append(rez)
        else:
            rez_bez.append(rez)

        # GRAFIKON 1: Stvarno vs Predviđeno
        naziv_safe = naziv_skupa.replace(' ', '_').replace('/', '')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Stvarne vs Predviđene vrednosti — {naziv_skupa}',
                     fontsize=12, fontweight='bold')

        boje = ['#4A90D9', '#2ECC71', '#E85D75']
        for ax, (naziv, yp), boja in zip(axes, default_predikcije.items(), boje):
            ax.scatter(y_test, yp, alpha=0.5, color=boja, edgecolors='white', s=40)
            lims = [min(y_test.min(), yp.min())-0.5, max(y_test.max(), yp.max())+0.5]
            ax.plot(lims, lims, 'k--', lw=1.2, label='Idealno')
            ax.set_xlabel('Stvarna ocena G3')
            ax.set_ylabel('Predviđena ocena G3')
            ax.set_title(naziv)
            ax.legend(fontsize=8)
            r2 = r2_score(y_test, yp)
            ax.text(0.05, 0.92, f'R²={r2:.3f}', transform=ax.transAxes,
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                    facecolor='white', alpha=0.7))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'stvarno_vs_predvidjeno_{naziv_safe}.png'),
                    bbox_inches='tight')
        plt.close()

        # GRAFIKON 2: Važnost atributa (samo bez G1/G2)
        if naziv_skupa == 'Bez G1 i G2':
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle('Važnost atributa — Bez G1 i G2 (tuned modeli)',
                         fontsize=12, fontweight='bold')

            for ax, (naziv_m, model) in zip(axes, [
                ('Slučajna šuma (tuned)',      gs_forest.best_estimator_),
                ('Stablo odlučivanja (tuned)', gs_tree.best_estimator_),
            ]):
                imp = model.feature_importances_
                df_imp = pd.DataFrame({'Atribut': X.columns, 'Važnost': imp})
                df_imp = df_imp.sort_values('Važnost', ascending=True).tail(15)
                boje_bar = ['#4A90D9' if v > 0.05 else '#AEC6E8' for v in df_imp['Važnost']]
                ax.barh(df_imp['Atribut'], df_imp['Važnost'], color=boje_bar, edgecolor='white')
                ax.set_title(naziv_m)
                ax.set_xlabel('Važnost (Feature Importance)')
                ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'vaznost_atributa_bez_G1_G2.png'),
                        bbox_inches='tight')
            plt.close()

    # GRAFIKON 3: Poređenje modela sa vs bez
    df_sa  = pd.DataFrame(rez_sa)
    df_bez = pd.DataFrame(rez_bez)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Poređenje modela — Sa G1/G2 vs Bez G1/G2',
                 fontsize=13, fontweight='bold')

    for ax, metrika, naslov in zip(axes,
        ['MAE', 'RMSE', 'R2'],
        ['MAE (niže = bolje)', 'RMSE (niže = bolje)', 'R² (više = bolje)']):

        x = np.arange(len(df_sa))
        w = 0.35
        bars1 = ax.bar(x - w/2, df_sa[metrika],  w, label='Sa G1/G2',
                       color='#4A90D9', edgecolor='white')
        bars2 = ax.bar(x + w/2, df_bez[metrika], w, label='Bez G1/G2',
                       color='#E85D75', edgecolor='white')

        for bar in list(bars1) + list(bars2):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}',
                    ha='center', va='bottom', fontsize=7)

        ax.set_title(naslov)
        ax.set_xticks(x)
        ax.set_xticklabels(df_sa['Model'], rotation=15, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'poredjenje_modela.png'), bbox_inches='tight')
    plt.close()

    # Rezime
    print("\nREZIME REZULTATA")
    print("\nSa G1 i G2:")
    print(df_sa.to_string(index=False))
    print("\nBez G1 i G2:")
    print(df_bez.to_string(index=False))

   
if __name__ == "__main__":
    pokreni_treniranje(PROCESSED_PATH, OUTPUT_DIR)