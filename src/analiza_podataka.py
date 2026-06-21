import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')


def nacrtaj_atribute_naspram_g3(df, output_dir):
    print("\nVizualizacija pojedinačnih atributa naspram G3...")

    atributi_za_prikaz = ['sex', 'address', 'Medu', 'studytime',
                           'failures', 'goout', 'romantic', 'higher']

    # Filtriramo samo atribute koji zaista postoje u datasetu (sigurnosna provera)
    atributi_za_prikaz = [a for a in atributi_za_prikaz if a in df.columns]

    n = len(atributi_za_prikaz)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for ax, atribut in zip(axes, atributi_za_prikaz):
        sns.boxplot(data=df, x=atribut, y='G3', ax=ax, color='#4A90D9')
        ax.set_title(f'G3 po atributu: {atribut}')
        ax.set_xlabel(atribut)
        ax.set_ylabel('G3' if atribut == atributi_za_prikaz[0] else '')
        ax.grid(axis='y', alpha=0.3)

    # Gasimo prazne panele ako broj atributa nije deljiv sa ncols
    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle('Odnos pojedinačnih atributa i završne ocene (G3)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, 'atributi_vs_g3.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    # Kratak tekstualni ispis prosečnog G3 po grupama, za svaki atribut
    #print("\nProsečna ocena G3 po grupama (za atribute prikazane na grafikonu):")
    #for atribut in atributi_za_prikaz:
    #    prosek_po_grupi = df.groupby(atribut)['G3'].mean().round(2)
    #    print(f"\n  {atribut}:")
    #    print(prosek_po_grupi.to_string())


def pokreni_eksplorativnu_analizu(data_path, output_dir):
    print("Eksplorativna analiza skupa")

    # Učitavamo sačuvani procesirani dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Greška: Nema podataka na {data_path}. Pokreni prvo priprema_podataka.py!")

    df = pd.read_csv(data_path)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Ispitivanje korelacija (Heatmap)
    print("\nRačunanje korelacija i iscrtavanje Heatmap-a...")

    # Biramo ključne numeričke atribute da grafikon ne bude prenatrpan i nepregledan
    kljucni_atributi = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                        'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']

    matrica_korelacije = df[kljucni_atributi].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(matrica_korelacije, annot=True, cmap='bwr', fmt=".2f", linewidths=0.5)
    plt.title('Matrica korelacije ključnih atributa sa završnom ocenom G3')

    # Čuvamo grafikon
    heatmap_path = os.path.join(output_dir, 'matrica_korelacije.png')
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()

    # Ispis najjačih korelacija sa G3 u terminalu
    print("\nTop 5 atributa koji imaju najjacu korelaciju sa završnom ocenom G3:")
    korelacij_sa_g3 = matrica_korelacije['G3'].drop(['G1', 'G2', 'G3']).sort_values(ascending=False)
    print(korelacij_sa_g3.head(5))
    print("\nAtributi sa najjačom NEGATIVNOM korelacijom (smanjuju ocenu):")
    print(korelacij_sa_g3.tail(2))

    # 2. Detekcija anomalija i ekstrema (Boxplot)
    print("\nAnaliza ekstremnih vrednosti (Outliers)")

    # Granica anomalije: 3 standardne devijacije od proseka (isti kriterijum
    # koristimo i za grafikon i za ispis, da se podaci poklapaju)
    granica_izostanaka = df['absences'].mean() + 3 * df['absences'].std()
    ekstremni_izostanci = df[df['absences'] > granica_izostanaka]

    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df['absences'], color='salmon')
    plt.axvline(granica_izostanaka, color='darkred', linestyle='--', linewidth=1.5,
                label=f'Granica anomalije (mean + 3·std = {granica_izostanaka:.1f})')
    plt.scatter(ekstremni_izostanci['absences'],
                np.zeros(len(ekstremni_izostanci)),
                color='black', marker='x', s=80, zorder=5,
                label=f'Ekstremne vrednosti (n={len(ekstremni_izostanci)})')
    plt.title('Detekcija ekstremnih vrednosti u broju izostanaka (Absences)')
    plt.xlabel('Broj izostanaka')
    plt.legend()

    boxplot_path = os.path.join(output_dir, 'izostanci_anomalije.png')
    plt.savefig(boxplot_path, bbox_inches='tight')
    plt.close()

    print(f"Broj učenika sa ekstremnim brojem izostanaka (preko {granica_izostanaka:.1f}): {len(ekstremni_izostanci)}")

    # 3. Odnos pojedinačnih atributa i G3
    nacrtaj_atribute_naspram_g3(df, output_dir)


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'student_processed.csv')
    OUTPUT_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'izvestaj_grafikoni')

    pokreni_eksplorativnu_analizu(PROCESSED_DATA_PATH, OUTPUT_IMAGES_DIR)