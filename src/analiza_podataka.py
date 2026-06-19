import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    korelacij_sa_g3 = matrica_korelacije['G3'].sort_values(ascending=False)
    print(korelacij_sa_g3.head(5)) # Prikazuje G3, G2, G1
    print("\nAtributi sa najjačom NEGATIVNOM korelacijom (smanjuju ocenu):")
    print(korelacij_sa_g3.tail(2)) 
    
    # 2. Detekcija anomalija i ekstrema (Boxplot)
    print("\nAnaliza ekstremnih vrednosti (Outliers)")
    
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df['absences'], color='salmon')
    plt.title('Detekcija ekstremnih vrednosti u broju izostanaka (Absences)')
    plt.xlabel('Broj izostanaka')
    
    boxplot_path = os.path.join(output_dir, 'izostanci_anomalije.png')
    plt.savefig(boxplot_path, bbox_inches='tight')
    plt.close()
    
    # Pronalaženje učenika sa ekstremnim brojem izostanaka (npr. preko 3 standardne devijacije)
    granica_izostanaka = df['absences'].mean() + 3 * df['absences'].std()
    ekstremni_izostanci = df[df['absences'] > granica_izostanaka]
    print(f"Broj učenika sa ekstremnim brojem izostanaka (preko {int(granica_izostanaka)}): {len(ekstremni_izostanci)}")

    # 3. Zakljcak
    #1. Atributi G1 i G2 imaju ekstremno visoku korelaciju sa G3 to znači da ako ih ostavimo, model će gledati SAMO njih, a ignorisaće sve ostalo.")
    #2. Odluka o obradi: Podatke u Fazi 3 delimo na dva skupa (sa i bez G1/G2) kako bismo naterali algoritam da uči iz demografskih i socijalnih faktora.")

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'student_processed.csv')
    OUTPUT_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'izvestaj_grafikoni')
    
    pokreni_eksplorativnu_analizu(PROCESSED_DATA_PATH, OUTPUT_IMAGES_DIR)