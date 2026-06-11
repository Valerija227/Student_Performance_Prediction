import os
import pandas as pd
import numpy as np

def prepare_data(input_path, output_path):
    print("Pokretanje faze pripreme podataka")
    
    # 1. Učitavanje podataka
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Greška: Fajl nije pronađen na putanji {input_path}")
        
    df = pd.read_csv(input_path)
    print(f"Uspešno učitan dataset. Dimenzije: {df.shape[0]} redova, {df.shape[1]} kolona.")

    # 2. Provera nedostajućih vrednosti (Missing values)
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    print("\nNedostajuće vrednosti:")
    if total_missing > 0:
        print(missing_values[missing_values > 0])
            #Moj dataset je cist
    else:
        print("Nema nedostajućih vrednosti u datasetu.")

    # 3. Provera anomalija (Deskriptivna statistika)
    print("\nDetekcija anomalija u ocenama i godinama:")
    anomalous_age = df[(df['age'] < 10) | (df['age'] > 25)]
    anomalous_grades = df[(df['G1'] < 0) | (df['G3'] > 20)]
    
    if len(anomalous_age) == 0 and len(anomalous_grades) == 0:
        print("Nisu uočene anomalije.")
    else:
        print(f"UPOZORENJE: Pronađeno {len(anomalous_age)} sumnjivih godina i {len(anomalous_grades)} sumnjivih ocena.")

    # 4. Enkodiranje kategorijskih podataka
# 4. Enkodiranje kategorijskih podataka
# 4. Enkodiranje kategorijskih podataka
    df_encoded = df.copy()
    
    # Prvo sve kolone koje sadrže tekst ručno prebacujemo u string tip podataka
    # i čistimo eventualne navodnike koji mogu da zbune Pandas
    for col in df_encoded.columns:
        # Ako je prvi element u koloni tekst (string), obeležavamo je kao tekstualnu
        if isinstance(df_encoded[col].iloc[0], str):
            df_encoded[col] = df_encoded[col].astype(str).str.replace('"', '').str.strip()

    # Sada definišemo tačan spisak kolona koje su kategorijske na osnovu opisa tvog projekta
    all_categorical = [
        'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
        'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
        'nursery', 'higher', 'internet', 'romantic'
    ]
    
    # Razdvajamo ih na binarne (2 opcije) i višestruke (3 ili više opcija)
    binary_cols = [col for col in all_categorical if col in df_encoded.columns and df_encoded[col].nunique() == 2]
    multi_cols = [col for col in all_categorical if col in df_encoded.columns and df_encoded[col].nunique() > 2]
    
    print("\nTransformacija tekstualnih podataka u brojeve:")
    print(f" Binarne kolone (Label Encoding): {binary_cols}")
    print(f" Višestruke kategorijske kolone (One-Hot Encoding): {multi_cols}")
    
    # A) Label Encoding za binarne (0 i 1)
    for col in binary_cols:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
        
    # B) One-Hot Encoding za višestruke kategorije
    if len(multi_cols) > 0:
        df_encoded = pd.get_dummies(df_encoded, columns=multi_cols, drop_first=True)
    
    # Konvertujemo True/False u 1/0 za novije verzije Pythona
    bool_cols = df_encoded.select_dtypes(include=['bool']).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    print(f"Enkodiranje završeno. Nove dimenzije dataseta: {df_encoded.shape[0]} redova, {df_encoded.shape[1]} kolona.")

    # 5. Čuvanje procesuiranih podataka    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_encoded.to_csv(output_path, index=False)
    print(f"\n")

if __name__ == "__main__":
    # Definišemo relativne putanje prateći tvoju strukturu projekta
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    RAW_DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'raw', 'student-por.csv')
    PROCESSED_DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'student_processed.csv')
    
    # Pokrećemo funkciju
    prepare_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)