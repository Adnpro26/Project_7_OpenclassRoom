# API de Prédiction de Score de Crédit

## Objectif du Projet

Cette API est conçue pour fournir un score de crédit basé sur un modèle de machine learning. Elle utilise Flask pour exposer un service RESTful qui permet aux utilisateurs de soumettre des données et de recevoir des prédictions en temps réel. Ce projet est destiné à aider les institutions financières à évaluer le risque de crédit des clients potentiels de manière automatisée et efficace.

## Structure des Dossiers

├── app.py # Fichier principal de l'API Flask 
├── best_model_parametersl.pkl # Modèle de machine learning pré-entraîné 
├── requirements.txt # Liste des dépendances du projet 
├── Procfile # Fichier de configuration pour le déploiement sur Heroku 
├── README.md # Description du projet et instructions 
├── app.py # script permet d'envoyez une requête POST avec les données des caractéristiques au format JSON à l'endpoint 
├── request.py # utilisé pour répondre à l'API en envoyant une requête POST avec des données
├── Streamtest_Api.py # déploiement de l'api via streamlit
├── .gitignore #ndiquer les fichiers et dossiers que Git doit ignorer et ne pas suivre dans le contrôle de version


# ===================================================
# Script complet : génération reporting Excel
# ===================================================

import pandas as pd
from openpyxl import load_workbook

# ===================================================
# 1️⃣ Chargement des données
# ===================================================
# Exemple : DataFrame avec sources A et B


# ===================================================
# 2️⃣ Construction de base_index
# ===================================================
def build_base_index(df):
    base_index = {}
    for _, row in df.iterrows():
        key = (
            row["periode"],
            row["indicateur"],
            row["pays"],
            row["activite"],
            row["source"]
        )
        base_index[key] = base_index.get(key, 0) + row["valeur"]
    return base_index

base_index = build_base_index(df)

# ===================================================
# 3️⃣ Définition des fonctions de lecture
# ===================================================
def read_source(base_index, business_key, source):
    return base_index.get((*business_key, source), 0)

def read_operational(base_index, business_key):
    a = read_source(base_index, business_key, "A")
    b = read_source(base_index, business_key, "B")
    return a - b

VISION_REGISTRY = {
    "A": lambda idx, key: read_source(idx, key, "A"),
    "B": lambda idx, key: read_source(idx, key, "B"),
    "OPERATIONAL": read_operational,
}

def read_value(base_index, business_key, vision):
    if vision not in VISION_REGISTRY:
        raise ValueError(f"Vision inconnue : {vision}")
    return VISION_REGISTRY[vision](base_index, business_key)

# ===================================================
# 4️⃣ Chargement du reporting Excel
# ===================================================
wb = load_workbook("reporting.xlsx")
ws = wb.active

# ===================================================
# 5️⃣ Fonctions utilitaires pour le reporting
# ===================================================
def get_period(ws):
    return ws["C3"].value

def get_indicator(ws, col):
    return ws.cell(row=13, column=col).value

def get_country(ws, row):
    return ws.cell(row=row, column=3).value  # colonne C

# ===================================================
# 6️⃣ Parcours du reporting et génération des valeurs
# ===================================================
START_ROW = 14
START_COL = 4  # colonne D
END_ROW = ws.max_row
END_COL = ws.max_column

period = get_period(ws)
vision = "OPERATIONAL"

for row in range(START_ROW, END_ROW + 1):
    country = get_country(ws, row)
    if not country:
        continue

    for col in range(START_COL, END_COL + 1):
        indicator = get_indicator(ws, col)
        if not indicator:
            continue

        business_key = (
            period,
            indicator,
            country,
            "SAVING"  # ou détecté dynamiquement
        )

        value = read_value(base_index, business_key, vision)

        ws.cell(row=row, column=col, value=value)

# ===================================================
# 7️⃣ Sauvegarde du reporting mis à jour
# ===================================================
wb.save("reporting_generated.xlsx")
print("✅ Reporting généré avec succès : reporting_generated.xlsx")


def compute_raw_indicator_df(df, indicator_name, cfg):
    mask = pd.Series(True, index=df.index)

    for col, value in cfg["filters"].items():
        mask &= df[col] == value

    filtered = df.loc[mask]

    if filtered.empty:
        return None

    return (
        filtered
        .groupby(["periode", "pays"], as_index=False)["valeur"]
        .sum()
        .assign(
            indicateur=indicator_name,
            activite=cfg["filters"]["activite"],
            source="computed"
        )
    )


cfg = yaml_cfg["indicators"]["ISR_Protection_Brut"]

df_isr_protection = compute_raw_indicator_df(
    df,
    "ISR_Protection_Brut",
    cfg
)

df = pd.concat([df, df_isr_protection], ignore_index=True)
