
import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="MBRL Education Spending Assessment",
    layout="wide"
)

st.title("MBRL Education Spending Assessment")

st.write(
    "Upload World Bank education expenditure data → "
    "MBRL-derived % GDP → Deviation → Spending State"
)


# ============================================================
# SIDEBAR PARAMETERS
# ============================================================

st.sidebar.header("MBRL Parameters")

# Fixed operational threshold = 10%
TAU = 0.10

st.sidebar.success("Classification threshold: ±10%")

trees = st.sidebar.slider(
    "World-model trees",
    min_value=100,
    max_value=800,
    value=400,
    step=50
)

st.sidebar.write("τ = 10%")


# ============================================================
# FILE UPLOAD
# ============================================================

up = st.file_uploader(
    "Upload Top 5 African Countries Excel Dataset",
    type=["xlsx"]
)


# ============================================================
# COLUMN ALIASES
# ============================================================

ALIASES = {

    "Country": [
        "Country",
        "Country Name",
        "country",
        "geoUnit"
    ],

    "Year": [
        "Year",
        "year"
    ],

    "Actual": [
        "Actual WB % GDP",
        "Actual WB %GDP",
        "Actual_WB_GDP",
        "WB % GDP",
        "Education % GDP",
        "Education expenditure % GDP",
        "Government expenditure on education, total (% of GDP)",
        "SE.XPD.TOTL.GD.ZS"
    ],

    "GDP": [
        "GDP",
        "GDP (current US$)",
        "NY.GDP.MKTP.CD"
    ],

    "GDP_Growth": [
        "GDP Growth",
        "GDP growth (annual %)",
        "NY.GDP.MKTP.KD.ZG"
    ],

    "GDP_pc": [
        "GDP per capita",
        "GDP per capita (current US$)",
        "NY.GDP.PCAP.CD"
    ],

    "Population": [
        "Population",
        "SP.POP.TOTL"
    ],

    "EduGov": [
        "Education % Government Expenditure",
        "Education expenditure % government expenditure",
        "SE.XPD.TOTL.GB.ZS"
    ]
}


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(df, possible_names):

    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in possible_names:

        key = name.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


# ============================================================
# INSTRUCTIONS BEFORE UPLOAD
# ============================================================

if up is None:

    st.info(
        "Upload an Excel file containing at least: "
        "Country, Year and Actual WB % GDP."
    )

    example = pd.DataFrame({

        "Country": [
            "South Africa",
            "South Africa",
            "Egypt",
            "Egypt",
            "Nigeria",
            "Nigeria"
        ],

        "Year": [
            2020,
            2021,
            2020,
            2021,
            2020,
            2021
        ],

        "Actual WB % GDP": [
            6.20,
            5.90,
            4.80,
            4.50,
            1.70,
            1.90
        ]
    })

    st.subheader("Expected Excel structure")

    st.dataframe(
        example,
        use_container_width=True,
        hide_index=True
    )

    st.stop()


# ============================================================
# READ EXCEL
# ============================================================

try:

    raw = pd.read_excel(up)

except Exception as e:

    st.error(f"Could not read Excel file: {e}")
    st.stop()


st.success(
    f"Excel uploaded successfully: {len(raw):,} rows"
)


# ============================================================
# DISPLAY ORIGINAL COLUMNS
# ============================================================

with st.expander("Uploaded Excel columns"):

    st.write(list(raw.columns))


# ============================================================
# IDENTIFY REQUIRED COLUMNS
# ============================================================

country_col = find_column(
    raw,
    ALIASES["Country"]
)

year_col = find_column(
    raw,
    ALIASES["Year"]
)

actual_col = find_column(
    raw,
    ALIASES["Actual"]
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing = []

if country_col is None:
    missing.append("Country")

if year_col is None:
    missing.append("Year")

if actual_col is None:
    missing.append("Actual WB % GDP")


if missing:

    st.error(
        "The following required columns were not detected: "
        + ", ".join(missing)
    )

    st.write("Columns detected in your Excel file:")

    st.write(list(raw.columns))

    st.stop()


# ============================================================
# STANDARDISE DATASET
# ============================================================

df = raw.rename(
    columns={
        country_col: "Country",
        year_col: "Year",
        actual_col: "Actual_WB_GDP"
    }
).copy()


# ============================================================
# OPTIONAL VARIABLES
# ============================================================

for variable in [
    "GDP",
    "GDP_Growth",
    "GDP_pc",
    "Population",
    "EduGov"
]:

    c = find_column(
        df,
        ALIASES[variable]
    )

    if c is not None and c != variable:

        df.rename(
            columns={c: variable},
            inplace=True
        )


# ============================================================
# CLEAN DATA
# ============================================================

df["Country"] = (
    df["Country"]
    .astype(str)
    .str.strip()
)

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

df["Actual_WB_GDP"] = pd.to_numeric(
    df["Actual_WB_GDP"],
    errors="coerce"
)


for c in [
    "GDP",
    "GDP_Growth",
    "GDP_pc",
    "Population",
    "EduGov"
]:

    if c in df.columns:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )


# ============================================================
# REMOVE INVALID YEARS
# ============================================================

df = df[
    df["Year"].notna()
].copy()

df["Year"] = df["Year"].astype(int)


# ============================================================
# SORT
# ============================================================

df = (
    df
    .sort_values(
        ["Country", "Year"]
    )
    .reset_index(drop=True)
)


# ============================================================
# SHOW DATA SUMMARY
# ============================================================

countries = df["Country"].nunique()

years = df["Year"].nunique()

valid_actual = df["Actual_WB_GDP"].notna().sum()

st.info(
    f"Countries: {countries} | "
    f"Years: {years} | "
    f"Valid education observations: {valid_actual}"
)


# ============================================================
# CREATE TEMPORAL FEATURES
# ============================================================

group = df.groupby(
    "Country",
    group_keys=False
)


df["Lag1"] = (
    group["Actual_WB_GDP"]
    .shift(1)
)

df["Lag2"] = (
    group["Actual_WB_GDP"]
    .shift(2)
)


df["Roll3"] = (
    group["Actual_WB_GDP"]
    .transform(
        lambda s:
        s.shift(1)
        .rolling(
            window=3,
            min_periods=1
        )
        .mean()
    )
)


year_min = df["Year"].min()

year_max = df["Year"].max()

year_range = max(
    1,
    year_max - year_min
)


df["YearNorm"] = (
    (df["Year"] - year_min)
    / year_range
)


# ============================================================
# OPTIONAL FEATURES
# ============================================================

extra_features = [

    c for c in [
        "GDP",
        "GDP_Growth",
        "GDP_pc",
        "Population",
        "EduGov"
    ]

    if c in df.columns
]


# ============================================================
# WORLD MODEL FEATURES
# ============================================================

features = [
    "Lag1",
    "Lag2",
    "Roll3",
    "YearNorm",
    "Action"
] + extra_features


# ============================================================
# BUILD TRANSITION DATA
#
# State_t + Action_t → Education spending at t+1
# ============================================================

transitions = []


for country, x in df.groupby("Country"):

    x = (
        x
        .sort_values("Year")
        .reset_index(drop=True)
    )

    for i in range(len(x) - 1):

        current = x.iloc[i]

        nxt = x.iloc[i + 1]

        # Both current and next observations
        # must contain actual WB values
        if pd.isna(current["Actual_WB_GDP"]):
            continue

        if pd.isna(nxt["Actual_WB_GDP"]):
            continue


        row = {}

        for f in features:

            if f == "Action":

                row[f] = current["Actual_WB_GDP"]

            else:

                row[f] = current.get(
                    f,
                    np.nan
                )


        row["Target"] = nxt["Actual_WB_GDP"]

        transitions.append(row)


# ============================================================
# TRANSITION DATAFRAME
# ============================================================

trans = pd.DataFrame(
    transitions
)


# ============================================================
# CHECK TRANSITIONS
# ============================================================

if len(trans) < 5:

    st.error(
        f"Only {len(trans)} usable transitions were created. "
        "At least 5 are required to train the world model."
    )

    st.write(
        "For best results use Top 5 African countries "
        "with annual observations from 2015–2025."
    )

    st.stop()


st.success(
    f"World model training transitions: {len(trans)}"
)


# ============================================================
# PREPARE MODEL DATA
# ============================================================

X = trans[features].copy()

y = trans["Target"].copy()


X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# IMPUTE MODEL FEATURES
# ============================================================

medians = X.median(
    numeric_only=True
)


X = X.fillna(
    medians
)


X = X.fillna(0)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

split = int(
    len(X) * 0.80
)


split = max(
    1,
    min(
        split,
        len(X) - 1
    )
)


X_train = X.iloc[:split]

X_test = X.iloc[split:]

y_train = y.iloc[:split]

y_test = y.iloc[split:]


# ============================================================
# RANDOM FOREST WORLD MODEL
# ============================================================

model = RandomForestRegressor(

    n_estimators=trees,

    random_state=42,

    min_samples_leaf=1,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION
# ============================================================

pred = model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    pred
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)


if len(y_test) > 1:

    r2 = r2_score(
        y_test,
        pred
    )

else:

    r2 = np.nan


# ============================================================
# MBRL POLICY
#
# Search for an education-spending target
# that gives the lowest model-based cost.
# ============================================================

candidate_actions = np.arange(
    0.5,
    12.01,
    0.1
)


def calculate_mbrl_policy(row):

    # --------------------------------------------------------
    # Estimate current state
    # --------------------------------------------------------

    if pd.notna(row["Lag1"]):

        lag = row["Lag1"]

    elif pd.notna(row["Roll3"]):

        lag = row["Roll3"]

    elif pd.notna(row["Actual_WB_GDP"]):

        lag = r
```
