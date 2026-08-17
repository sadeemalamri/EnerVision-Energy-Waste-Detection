import streamlit as st
import pandas as pd
import numpy as np
import joblib

# PAGE CONFIG
st.set_page_config(
    page_title="Energy Waste Detection",
    page_icon="⚡",
    layout="centered"
)

st.title("Industrial Energy Waste Detection System")
st.write("Enter the facility information to estimate whether energy usage is normal or potentially wasteful.")


# LOAD MODEL + SCALER + DEFAULTS
MODELS_DIR = "models"
model = joblib.load(f"{MODELS_DIR}/isolation_forest_model.pkl")
scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
fuel_defaults = joblib.load(f"{MODELS_DIR}/fuel_defaults.pkl")   # {fuel: {MMBtu: x, GWht: y}}


# FEATURE LIST
all_features = [
 'FACILITY_ID',
 'REPORTING_YEAR',
 'PRIMARY_NAICS_CODE',
 'COGENERATION_UNIT_EMISS_IND',
 'MMBtu_TOTAL',
 'GWht_TOTAL',
 'fuel_Agricultural Byproducts',
 'fuel_Anthracite',
 'fuel_Biodiesel (100%)',
 'fuel_Bituminous',
 'fuel_Blast Furnace Gas',
 'fuel_Butane',
 'fuel_Coal Coke',
 'fuel_Coke Oven Gas',
 'fuel_Distillate Fuel Oil No. 1',
 'fuel_Distillate Fuel Oil No. 2',
 'fuel_Distillate Fuel Oil No. 4',
 'fuel_Ethane',
 'fuel_Ethanol (100%)',
 'fuel_Ethylene',
 'fuel_Fuel Gas',
 'fuel_Heavy Gas Oils',
 'fuel_Kerosene',
 'fuel_Kerosene-Type Jet Fuel',
 'fuel_Landfill Gas',
 'fuel_Lignite',
 'fuel_Liquefied petroleum gases (LPG)',
 'fuel_Lubricants',
 'fuel_Mixed (Commercial sector)',
 'fuel_Mixed (Electric Power sector)',
 'fuel_Mixed (Industrial coking)',
 'fuel_Mixed (Industrial sector)',
 'fuel_Motor Gasoline',
 'fuel_Municipal Solid Waste',
 'fuel_Naphtha (<401 deg F)',
 'fuel_Natural Gas (Weighted U.S. Average)',
 'fuel_Natural Gasoline',
 'fuel_Other Biomass Gases',
 'fuel_Other Oil (>401 deg F)',
 'fuel_Pentanes Plus',
 'fuel_Petroleum Coke',
 'fuel_Plastics',
 'fuel_Propane',
 'fuel_Propane Gas',
 'fuel_Propylene',
 'fuel_Rendered Animal Fat',
 'fuel_Residual Fuel Oil No. 5',
 'fuel_Residual Fuel Oil No. 6',
 'fuel_Solid Byproducts',
 'fuel_Special Naphtha',
 'fuel_Subbituminous',
 'fuel_Tires',
 'fuel_Unfinished Oils',
 'fuel_Used Oil',
 'fuel_Vegetable Oil',
 'fuel_Wood and Wood Residuals (dry basis)'
]

fuel_cols = [c for c in all_features if c.startswith("fuel_")]
fuel_labels = [c.replace("fuel_", "") for c in fuel_cols]


# FUEL TYPE
st.subheader("Fuel Type")
fuel_choice_label = st.selectbox(
    "Select the primary fuel type:",
    fuel_labels
)
fuel_choice_col = "fuel_" + fuel_choice_label

# SMART DEFAULTS BASED ON FUEL TYPE
default_MMBtu = float(fuel_defaults[fuel_choice_label]["MMBtu"])
default_GWht  = float(fuel_defaults[fuel_choice_label]["GWht"])


# INDUSTRY TYPE (NAICS)
industry_options = {
    "Unknown / Not Sure": 0,
    "Food Manufacturing": 311,
    "Chemical Manufacturing": 325,
    "Primary Metal Manufacturing": 331,
    "Paper Manufacturing": 322,
    "Petroleum & Coal Products": 324,
    "Nonmetallic Minerals (Cement/Glass)": 327,
    "Fabricated Metal": 332,
    "Plastics & Rubber Products": 326,
    "Transportation Equipment": 336,
}

st.subheader("Industry Type")
industry_choice = st.selectbox("Select the facility’s industry:", list(industry_options.keys()))
NAICS = industry_options[industry_choice]


# REPORTING YEAR
st.subheader("Reporting Year")
REPORTING_YEAR = st.number_input(
    "Enter reporting year:",
    min_value=1900,
    max_value=2100,
    value=2014
)


# COGENERATION STATUS
st.subheader("Cogeneration Status")
cogen_label = st.selectbox("Does the facility use cogeneration units?", ["No", "Yes"])
cogen = 1 if cogen_label == "Yes" else 0


# ENERGY INPUTS (SMART DEFAULTS)
st.subheader("Energy Consumption Inputs")

MMBtu = st.number_input(
    "Total MMBtu (Thermal Energy Consumption):",
    min_value=0.0,
    value=default_MMBtu
)

GWht = st.number_input(
    "Total GWht (Electricity Consumption):",
    min_value=0.0,
    value=default_GWht
)


# BUILDING ROW FOR MODEL
row = pd.DataFrame(columns=all_features)
row.loc[0] = 0

# Adding values into row
row.at[0, "FACILITY_ID"] = 0
row.at[0, "REPORTING_YEAR"] = REPORTING_YEAR
row.at[0, "PRIMARY_NAICS_CODE"] = NAICS
row.at[0, "COGENERATION_UNIT_EMISS_IND"] = cogen
row.at[0, "MMBtu_TOTAL"] = MMBtu
row.at[0, "GWht_TOTAL"] = GWht

# set fuel type = 1
if fuel_choice_col in row.columns:
    row.at[0, fuel_choice_col] = 1


# RUN DETECTION
if st.button("Waste Detection"):
    scaled = scaler.transform(row)
    raw_pred = model.predict(scaled)[0]
    prediction = 1 if raw_pred == -1 else 0

    st.subheader("Result")
    if prediction == 1:
        st.error(" **Possible energy waste detected.**")
    else:
        st.success("**Energy usage appears normal.**")
