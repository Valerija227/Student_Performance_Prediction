# Student Performance Prediction

Predikcija završne ocene učenika (G3) na osnovu školskih i socio-demografskih podataka, korišćenjem algoritama mašinskog učenja. Projekat upoređuje dva scenarija — sa i bez dostupnih parcijalnih ocena (G1, G2) — kako bi se procenilo koliko je uspeh učenika predvidiv isključivo na osnovu demografskog i socijalnog profila.

## Sadržaj

- [Opis projekta](#opis-projekta)
- [Struktura foldera](#struktura-foldera)
- [Instalacija](#instalacija)
- [Pokretanje](#pokretanje)
- [Rezultati](#rezultati)

## Opis projekta

Dataset: [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) (UCI Machine Learning Repository), predmet portugalski jezik (`student-por.csv`).

Projekat obuhvata:

- pripremu i čišćenje podataka (enkodiranje, detekcija anomalija)
- eksplorativnu analizu (korelacije, raspodele, outlieri, odnos pojedinačnih atributa i G3)
- selekciju atributa (Feature Importance, SelectKBest, RFE)
- treniranje i poređenje tri modela (Linearna regresija, Stablo odlučivanja, Slučajna šuma) u default i podešenoj (tuned) varijanti
- evaluaciju kroz MAE, RMSE, R² i 5-fold unakrsnu validaciju
- Streamlit aplikaciju za interaktivnu predikciju

## Struktura foldera

```
Student_Performance_Prediction/
├── data/
│   ├── raw/                  # originalni dataset (student-por.csv)
│   └── processed/            # očišćen i enkodiran dataset
├── src/
│   ├── priprema_podataka.py  # čišćenje i enkodiranje
│   ├── analiza_podataka.py   # eksplorativna analiza (EDA)
│   ├── odabir_atributa.py    # selekcija atributa (3 metode)
│   ├── treniranje_modela.py  # treniranje, tuning, grafikoni za izveštaj
│   ├── model.py               # čuvanje finalnih modela (.pkl)
│   └── app.py                 # Streamlit aplikacija
├── modeli/                   # sačuvani .pkl modeli (generišu se)
├── izvestaj_grafikoni/       # generisani grafikoni (generišu se)
├── rezultati.txt              # tekstualni izveštaj rezultata (generiše se)
└── README.md
```

## Instalacija

Potreban je Python 3.10+.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

## Pokretanje

Skripte se pokreću ovim redosledom, iz `src/` foldera:

```bash
python priprema_podataka.py   # 1. čišćenje i enkodiranje podataka
python analiza_podataka.py    # 2. EDA grafikoni (korelacije, outlieri, atributi vs G3)
python odabir_atributa.py     # 3. selekcija atributa (ispisuje finalnu FEAT_BEZ listu)
python treniranje_modela.py   # 4. treniranje, tuning, grafikoni za izveštaj
python model.py                # 5. čuvanje finalnih modela za aplikaciju
python -m streamlit run app.py           # 6. pokretanje interaktivne aplikacije
```

Svaki korak zavisi od prethodnog (npr. `treniranje_modela.py` zahteva da je `priprema_podataka.py` već pokrenut) — ako neki fajl nedostaje, skripta će to javiti uz uputstvo šta prvo pokrenuti.

## Generisani grafikoni

Pokretanjem `analiza_podataka.py` i `treniranje_modela.py` u `izvestaj_grafikoni/` se generišu:

| Fajl | Opis |
|---|---|
| `matrica_korelacije.png` | Korelacija ključnih numeričkih atributa sa G3 |
| `izostanci_anomalije.png` | Detekcija ekstremnih vrednosti u broju izostanaka |
| `atributi_vs_g3.png` | Raspodela G3 po pojedinačnim atributima (sex, address, Medu, studytime, failures, goout, romantic, higher) |
| `odabir_atributa_metode_*.png` | Poređenje 3 metode selekcije atributa, za oba scenarija |
| `greska_po_opsegu_*.png` | Prosečna greška modela po opsegu ocena, za oba scenarija |
| `vaznost_atributa_bez_G1_G2.png` | Važnost atributa kod tuned modela (scenario bez G1/G2) |
| `poredjenje_modela.png` | Poređenje MAE/RMSE/R² svih modela, oba scenarija |

## Rezultati

| Scenario | Najbolji model | MAE | R² |
|---|---|---|---|
| Sa G1 i G2 | Linearna regresija | 0.727 | 0.899 |
| Bez G1 i G2 | Slučajna šuma (tuned) | 2.044 | 0.225 |

Detaljna analiza rezultata, metodologija i diskusija nalaze se u pratećem izveštaju projekta.