# Smart Energy AI — Energy Waste Detection in Industrial Facilities

An unsupervised machine learning system that detects abnormal / wasteful
energy consumption in industrial facilities, built with an **Isolation
Forest** model and served through an interactive **Streamlit** dashboard.

Developed as an AI Course Capstone Project (Samsung Innovation Campus) by
**Team EnerVision**.

## Overview

Industrial facilities consume large amounts of energy across different
fuel types and sectors. This project uses the *Industrial Energy End Use*
dataset to train an anomaly detection model that flags facilities whose
energy usage deviates significantly from expected patterns — helping
energy managers target audits and reduce inefficiencies.

## Project Structure

```
.
├── app.py                  # Streamlit dashboard / inference app
├── requirements.txt
├── models/
│   ├── isolation_forest_model.pkl   # trained Isolation Forest model
│   ├── scaler.pkl                   # fitted StandardScaler
│   └── fuel_defaults.pkl            # per-fuel-type average MMBtu/GWh (used for smart form defaults)
├── notebooks/
│   ├── DataAnalysis.ipynb           # exploratory data analysis (EDA)
│   └── Isolation_Forest.ipynb       # preprocessing + model training
└── data/
    ├── industrialcombenergy-2014.csv        # raw dataset
    └── clean_industrial_energy_encoded1.csv # cleaned & one-hot encoded dataset
```

## Methodology

1. **Data acquisition** — Industrial Energy End Use dataset (Kaggle).
2. **Preprocessing** — dropped irrelevant columns, mapped
   `COGENERATION_UNIT_EMISS_IND` to 0/1, one-hot encoded `FUEL_TYPE`.
3. **EDA** — distribution, correlation, and sector/fuel breakdown analysis
   (see `notebooks/DataAnalysis.ipynb`).
4. **Modeling** — `IsolationForest` (`n_estimators=300`,
   `contamination=0.05`) trained on scaled facility features to flag the
   ~5% most anomalous (potentially wasteful) facilities.
5. **Deployment** — the trained model, scaler, and per-fuel defaults are
   loaded into a Streamlit app (`app.py`) that lets a user enter facility
   details and get an instant waste/normal prediction.

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Team

EnerVision — Raghad Almadi, Sundos Alshutwi, Shahad Almowaled,
Sadeem Alamri, Taif Alharthi, Samiah Tolah.
