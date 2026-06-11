import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'student_processed.csv')
MODELS_DIR = os.path.join(CURRENT_DIR, 'modeli')
os.makedirs(MODELS_DIR, exist_ok=True)

# Selektovani atributi iz faze odabira atributa
FEAT_SA  = ['G2', 'G1', 'Medu', 'Dalc', 'reason_other']
FEAT_BEZ = ['failures', 'Dalc', 'Fedu', 'studytime', 'Walc',
            'higher', 'reason_other', 'reason_reputation']


def pokreni_export(data_path, models_dir):
    print("EXPORT MODELA")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Nema podataka na: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Učitan dataset: {df.shape[0]} redova, {df.shape[1]} kolona")

    y = df['G3']

    skupovi = [
        {
            'naziv':  'sa_G1G2',
            'X':      df[FEAT_SA],
            'model':  DecisionTreeRegressor(random_state=42),
            'params': {
                'max_depth':         [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf':  [1, 2, 4],
            }
        },
        {
            'naziv':  'bez_G1G2',
            'X':      df[FEAT_BEZ],
            'model':  RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators':      [100, 200],
                'max_depth':         [5, 10, None],
                'min_samples_split': [2, 5],
            }
        },
    ]

    for skup in skupovi:
        naziv = skup['naziv']
        X     = skup['X']
        print(f"\nTreniranje modela: {naziv}")
        print(f"  Atributi ({len(X.columns)}): {X.columns.tolist()}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        gs = GridSearchCV(skup['model'], skup['params'],
                          cv=5, scoring='r2', n_jobs=-1)
        gs.fit(X_train, y_train)

        best = gs.best_estimator_
        yp   = np.clip(np.round(best.predict(X_test)), 0, 20)

        mae  = mean_absolute_error(y_test, yp)
        rmse = np.sqrt(mean_squared_error(y_test, yp))
        r2   = r2_score(y_test, yp)

        print(f"  Najbolji parametri: {gs.best_params_}")
        print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

        paket = {
            'model':    best,
            'features': X.columns.tolist(),
            'metrics':  {'MAE': round(mae,3), 'RMSE': round(rmse,3), 'R2': round(r2,3)},
        }

        # ✅ ISPRAVKA: svaki model u svoj fajl
        path = os.path.join(models_dir, f'model_{naziv}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(paket, f)
       

   # print("\nExport završen!")
   # print(f"Modeli sačuvani u: {models_dir}")


if __name__ == "__main__":
    pokreni_export(PROCESSED_PATH, MODELS_DIR)