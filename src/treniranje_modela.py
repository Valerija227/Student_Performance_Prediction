import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
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

FEAT_BEZ = ['failures', 'higher', 'absences', 'studytime', 'Medu',
            'Walc', 'Dalc', 'school', 'Mjob_teacher']
FEAT_SA = FEAT_BEZ + ['G1', 'G2']

# Granice za grupisanje ocena pri analizi greske po opsegu
BINS_OCENA = [0, 9, 13, 20]
LABELE_OCENA = ['Niske (≤9)', 'Srednje (10–13)', 'Visoke (≥14)']

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
    finalni_model.fit(X_train, y_train)  # trenira se SAMO na Train, isto kao default modeli
    return finalni_model, najbolji_params, najbolji_mae


def nacrtaj_gresku_po_opsegu(y_test, default_predikcije, naziv_skupa, output_dir):
    grupe_test = pd.cut(y_test, bins=BINS_OCENA, labels=LABELE_OCENA, include_lowest=True)

    rezultati_po_grupi = {}
    for naziv, y_pred in default_predikcije.items():
        greske = np.abs(y_test.values - y_pred)
        df_g = pd.DataFrame({'grupa': grupe_test.values, 'greska': greske})
        rezultati_po_grupi[naziv] = df_g.groupby('grupa')['greska'].mean().reindex(LABELE_OCENA)

    df_rez = pd.DataFrame(rezultati_po_grupi)

    fig, ax = plt.subplots(figsize=(9, 5))
    df_rez.plot(kind='bar', ax=ax, color=['#4A90D9', '#2ECC71', '#E85D75'], edgecolor='white')
    ax.set_ylabel('Prosečna apsolutna greška (MAE)')
    ax.set_xlabel('Opseg stvarne ocene G3')
    ax.set_title(f'Prosečna greška po opsegu ocena — {naziv_skupa} (default modeli)')
    ax.legend(title='Model')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()

    naziv_safe = naziv_skupa.replace(' ', '_').replace('/', '')
    plt.savefig(os.path.join(output_dir, f'greska_po_opsegu_{naziv_safe}.png'), bbox_inches='tight')
    plt.close()


def pokreni_treniranje(data_path, output_dir, rezultati_path):
    print("TRENIRANJE I PODEŠAVANJE HIPERPARAMETARA")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Nema podataka na: {data_path}\nPokreni prvo priprema_podataka.py!")

    df = pd.read_csv(data_path)
    print(f"\nUčitan dataset: {df.shape[0]} redova, {df.shape[1]} kolona")

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
    log_linije.append("Svi modeli (default i tuned) treniraju se na IDENTIČNOJ količini")
    log_linije.append("podataka (Train, 70%) radi fer poređenja.")

    svi_rezultati = {}
    svi_cv_rezultati = {}

    for naziv_skupa, X in skupovi.items():
        print(f"\nSKUP: {naziv_skupa.upper()}")

        log_linije.append("")
        log_linije.append("=" * 56)
        log_linije.append(f"REZULTATI — {naziv_skupa.upper()}")
        log_linije.append("=" * 56)

        X_train, X_val, X_test, y_train, y_val, y_test = podeli_70_15_15(X, y)

        print(f"Podela: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
        log_linije.append(f"Podela: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

        # ── DEO A: Default modeli (treniraju se na Train, evaluiraju na Test) ──
        print("\nDEO A: Treniranje sa podrazumevanim parametrima")
        log_linije.append("")
        log_linije.append("DEO A: Default modeli")

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

        print("\nDEO A.1: Provera stabilnosti (5-fold CV na trening skupu) , ne kriterijum odabira")
        log_linije.append("")
        log_linije.append("DEO A.1: Provera stabilnosti (5-fold CV na trening skupu)")

        cv_rezultati_skupa = {}
        for naziv, model in default_modeli.items():
            cv_mae = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            cv_rezultati_skupa[naziv] = {'mean': cv_mae.mean(), 'std': cv_mae.std()}
            linija = (f"  {naziv:<30} CV-MAE prosek={cv_mae.mean():.3f}  "
                      f"std={cv_mae.std():.3f} )")
            print(linija)
            log_linije.append(linija)
        svi_cv_rezultati[naziv_skupa] = cv_rezultati_skupa

        # ── DEO B: Tuning (na Validation skupu) ──
        print("\nDEO B: Podešavanje hiperparametara (na Validation skupu)")
        log_linije.append("")
        log_linije.append("DEO B: Nakon tuninga (izbor hiperparametara na Validation skupu,")
        log_linije.append("        finalni model treniran samo na Train — isto kao default modeli)")

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

        # ── Grafikon: greska po opsegu ocena (zamena za scatter) ──
        nacrtaj_gresku_po_opsegu(y_test, default_predikcije, naziv_skupa, output_dir)

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
        log_linije.append(f"\nSkup: {naziv_skupa}")
        for naziv_m, m in rezultati_skupa.items():
            oznaka = "  <-- najbolji (najniži MAE na Test skupu)" if naziv_m == najbolji_naziv else ""
            log_linije.append(f"  {naziv_m:<30} MAE={m['MAE']}  RMSE={m['RMSE']}  R²={m['R2']}{oznaka}")

        # Dijagnostička napomena o stabilnosti, na osnovu CV-a iz Dela A.1
        log_linije.append("  Stabilnost (CV std na trening skupu, samo default modeli):")
        for naziv_m, cv in svi_cv_rezultati[naziv_skupa].items():
            napomena = "stabilno" if cv['std'] < 0.3 * cv['mean'] else "osetljivo na podelu podataka"
            log_linije.append(f"    {naziv_m:<28} std={cv['std']:.3f}  ({napomena})")

    with open(rezultati_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_linije))

if __name__ == "__main__":
    pokreni_treniranje(PROCESSED_PATH, OUTPUT_DIR, REZULTATI_PATH)