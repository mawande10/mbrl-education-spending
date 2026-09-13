import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MBRL Education Spending",
    layout="wide"
)

st.title("MBRL Education Spending Assessment")

st.write(
    "World Bank education expenditure → RF world model → "
    "model-based policy optimisation → MBRL-derived % GDP → "
    "Deviation → Spending State"
)

# ============================================================
# SIDEBAR PARAMETERS
# ============================================================

tau = st.sidebar.slider(
    "Threshold τ",
    min_value=0.01,
    max_value=0.30,
    value=0.10,
    step=0.01
)

trees = st.sidebar.slider(
    "World-model trees",
    min_value=100,
    max_value=800,
    value=400,
    step=50
)

min_spending = st.sidebar.number_input(
    "Minimum candidate spending (% GDP)",
    min_value=0.1,
    max_value=10.0,
    value=0.5,
    step=0.1
)

max_spending = st.sidebar.number_input(
    "Maximum candidate spending (% GDP)",
    min_value=1.0,
    max_value=20.0,
    value=12.0,
    step=0.5
)

candidate_step = st.sidebar.number_input(
    "Candidate step",
    min_value=0.01,
    max_value=1.0,
    value=0.1,
    step=0.01
)

# ============================================================
# FILE UPLOAD
# ============================================================

up = st.file_uploader(
    "Upload Excel workbook",
    type=["xlsx"]
)

# ============================================================
# COLUMN ALIASES
# ============================================================

ALIASES = {

    "Country": [
        "Country",
        "Country Name",
        "geoUnit"
    ],

    "Year": [
        "Year",
        "year"
    ],

    "Actual": [
        "Actual WB % GDP",
        "Actual WB %GDP",
        "WB % GDP",
        "Education % GDP",
        "Education expenditure % GDP",
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
        "GDP Per Capita",
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

def find_column(df, names):

    mapping = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in names:

        key = name.strip().lower()

        if key in mapping:
            return mapping[key]

    return None


# ============================================================
# WAIT FOR UPLOAD
# ============================================================

if up is None:

    st.info(
        "Required columns: Country, Year, Actual WB % GDP"
    )

    example = pd.DataFrame({
        "Country": [
            "Country A",
            "Country B",
            "Country C"
        ],

        "Year": [
            2022,
            2022,
            2022
        ],

        "Actual WB % GDP": [
            6.2,
            2.1,
            8.5
        ]
    })

    st.dataframe(
        example,
        use_container_width=True
    )

    st.stop()


# ============================================================
# READ EXCEL
# ============================================================

try:

    raw = pd.read_excel(up)

except Exception as e:

    st.error(
        f"Could not read the Excel workbook: {e}"
    )

    st.stop()


# ============================================================
# FIND REQUIRED COLUMNS
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


if any(
    c is None
    for c in [
        country_col,
        year_col,
        actual_col
    ]
):

    st.error(
        "Excel must contain columns equivalent to "
        "Country, Year and Actual WB % GDP."
    )

    st.write(
        "Detected columns:"
    )

    st.write(
        list(raw.columns)
    )

    st.stop()


# ============================================================
# STANDARDISE REQUIRED COLUMNS
# ============================================================

df = raw.rename(
    columns={
        country_col: "Country",
        year_col: "Year",
        actual_col: "Actual_WB_GDP"
    }
).copy()


# ============================================================
# OPTIONAL WORLD BANK VARIABLES
# ============================================================

for key in [
    "GDP",
    "GDP_Growth",
    "GDP_pc",
    "Population",
    "EduGov"
]:

    c = find_column(
        df,
        ALIASES[key]
    )

    if c is not None and c != key:

        df = df.rename(
            columns={
                c: key
            }
        )


# ============================================================
# DATA TYPES
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
# REMOVE INVALID COUNTRY/YEAR
# ============================================================

df = df[
    df["Country"].notna()
    & df["Year"].notna()
].copy()


# ============================================================
# SORT PANEL DATA
# ============================================================

df = df.sort_values(
    ["Country", "Year"]
).reset_index(drop=True)


# ============================================================
# CREATE LAG FEATURES
# ============================================================

grouped = df.groupby(
    "Country",
    group_keys=False
)

df["Lag1"] = grouped[
    "Actual_WB_GDP"
].shift(1)

df["Lag2"] = grouped[
    "Actual_WB_GDP"
].shift(2)

df["Roll3"] = grouped[
    "Actual_WB_GDP"
].transform(
    lambda s:
    s.shift(1)
    .rolling(
        window=3,
        min_periods=1
    )
    .mean()
)


# ============================================================
# NORMALISED YEAR
# ============================================================

year_min = df["Year"].min()
year_max = df["Year"].max()

df["YearNorm"] = (
    (df["Year"] - year_min)
    /
    max(
        1,
        year_max - year_min
    )
)


# ============================================================
# OPTIONAL FEATURES
# ============================================================

extra_features = [
    c
    for c in [
        "GDP",
        "GDP_Growth",
        "GDP_pc",
        "Population",
        "EduGov"
    ]
    if c in df.columns
]


# ============================================================
# MODEL FEATURES
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
# ============================================================

rows = []

for country, x in df.groupby("Country"):

    x = (
        x.sort_values("Year")
        .reset_index(drop=True)
    )

    for i in range(len(x) - 1):

        current = x.loc[i]
        nxt = x.loc[i + 1]

        # Need current and next actual values
        if (
            pd.isna(
                current["Actual_WB_GDP"]
            )
            or
            pd.isna(
                nxt["Actual_WB_GDP"]
            )
        ):

            continue

        record = {}

        for feature in features:

            if feature == "Action":

                record[feature] = (
                    current["Actual_WB_GDP"]
                )

            else:

                record[feature] = (
                    current[feature]
                )

        record["Target"] = (
            nxt["Actual_WB_GDP"]
        )

        rows.append(record)


trans = pd.DataFrame(rows)


# ============================================================
# CHECK TRANSITIONS
# ============================================================

if len(trans) < 5:

    st.error(
        "At least 5 historical transitions are "
        "required. Preferably upload data for "
        "54 African countries × 2015–2025."
    )

    st.stop()


# ============================================================
# PREPARE X AND Y
# ============================================================

X = trans[features].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Median imputation
for c in X.columns:

    if X[c].isna().any():

        median_value = X[c].median()

        if pd.isna(median_value):

            median_value = 0

        X[c] = X[c].fillna(
            median_value
        )


y = pd.to_numeric(
    trans["Target"],
    errors="coerce"
)


valid = y.notna()

X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)


# ============================================================
# TIME-ORDERED VALIDATION SPLIT
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
    min_samples_leaf=2,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# WORLD MODEL VALIDATION
# ============================================================

pred = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    pred
)

rmse = mean_squared_error(
    y_test,
    pred
) ** 0.5


if len(y_test) > 1:

    r2 = r2_score(
        y_test,
        pred
    )

else:

    r2 = np.nan


# ============================================================
# MBRL CANDIDATE ACTIONS
# ============================================================

cands = np.arange(
    min_spending,
    max_spending + candidate_step,
    candidate_step
)


# ============================================================
# WORLD MODEL IMPUTATION VALUES
# ============================================================

feature_medians = (
    X_train
    .median()
    .fillna(0)
)


# ============================================================
# MBRL POLICY
# ============================================================

def policy(row):

    # Previous education spending
    if pd.notna(row["Lag1"]):

        lag = row["Lag1"]

    elif pd.notna(row["Roll3"]):

        lag = row["Roll3"]

    elif pd.notna(row["Actual_WB_GDP"]):

        lag = row["Actual_WB_GDP"]

    else:

        lag = feature_medians.get(
            "Action",
            0
        )


    results = []

    for action in cands:

        candidate = {}

        for feature in features:

            if feature == "Action":

                candidate[feature] = action

            else:

                candidate[feature] = row.get(
                    feature,
                    np.nan
                )


        candidate_df = pd.DataFrame(
            [candidate],
            columns=features
        )

        candidate_df = candidate_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        candidate_df = candidate_df.fillna(
            feature_medians
        )

        candidate_df = candidate_df.fillna(0)


        # World model prediction
        predicted_next = model.predict(
            candidate_df
        )[0]


        # ----------------------------------------------------
        # MODEL-BASED COST FUNCTION
        # ----------------------------------------------------
        #
        # 1. Stability:
        #    avoid large discrepancy between
        #    predicted future expenditure and action
        #
        # 2. Smoothness:
        #    avoid sudden changes from previous spending
        #
        # 3. Policy preference:
        #    discourage values outside 4–6%
        #

        stability_cost = (
            predicted_next - action
        ) ** 2


        smoothness_cost = (
            action - lag
        ) ** 2


        policy_penalty = (
            max(0, 4 - action) ** 2
            +
            max(0, action - 6) ** 2
        )


        total_cost = (
            stability_cost
            +
            0.25 * smoothness_cost
            +
            0.10 * policy_penalty
        )


        results.append(
            (
                total_cost,
                action
            )
        )


    # Minimum-cost action
    return min(
        results,
        key=lambda x: x[0]
    )[1]


# ============================================================
# CALCULATE MBRL OUTPUT
# ============================================================

def calculate_result(row):

    # --------------------------------------------------------
    # MBRL POLICY
    # --------------------------------------------------------

    if pd.notna(
        row["Actual_WB_GDP"]
    ):

        mbrl_value = policy(row)

    else:

        # For missing observed data, still estimate
        # using the available state.
        mbrl_value = policy(row)


    # --------------------------------------------------------
    # DEVIATION
    # --------------------------------------------------------

    if (
        pd.notna(
            row["Actual_WB_GDP"]
        )
        and
        pd.notna(mbrl_value)
        and
        mbrl_value != 0
    ):

        deviation = (
            (
                row["Actual_WB_GDP"]
                -
                mbrl_value
            )
            /
            mbrl_value
        ) * 100

    else:

        deviation = np.nan


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if pd.isna(deviation):

        state = "Not classified"

    elif deviation < -tau * 100:

        state = "Underspending"

    elif deviation > tau * 100:

        state = "Overspending"

    else:

        state = "Normal Spending"


    return pd.Series(
        [
            mbrl_value,
            deviation,
            state
        ]
    )


# ============================================================
# APPLY MBRL
# ============================================================

df[
    [
        "MBRL_Derived_GDP",
        "Deviation",
        "State"
    ]
] = df.apply(
    calculate_result,
    axis=1
)


# ============================================================
# OUTPUT TABLE
# ============================================================

out = df[
    [
        "Country",
        "Year",
        "Actual_WB_GDP",
        "MBRL_Derived_GDP",
        "Deviation",
        "State"
    ]
].copy()


out.columns = [
    "Country",
    "Year",
    "Actual WB % GDP",
    "MBRL-derived % GDP",
    "Deviation",
    "State"
]


# ============================================================
# DISPLAY
# ============================================================

st.subheader(
    "Automated MBRL Results"
)


show = out.copy()


show[
    "Actual WB % GDP"
] = show[
    "Actual WB % GDP"
].round(2)


show[
    "MBRL-derived % GDP"
] = show[
    "MBRL-derived % GDP"
].round(2)


show[
    "Deviation"
] = show[
    "Deviation"
].map(
    lambda x:
    f"{x:+.2f}%"
    if pd.notna(x)
    else ""
)


st.dataframe(
    show,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SUMMARY METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)


m1.metric(
    "Records",
    len(out)
)


m2.metric(
    "MBRL mean",
    f"{out['MBRL-derived % GDP'].mean():.2f}%"
)


m3.metric(
    "Actual mean",
    f"{out['Actual WB % GDP'].mean():.2f}%"
)


m4.metric(
    "Threshold",
    f"±{tau:.0%}"
)


# ============================================================
# WORLD MODEL VALIDATION
# ============================================================

st.subheader(
    "World-model validation"
)


validation = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R²"
    ],

    "Value": [
        round(mae, 4),
        round(rmse, 4),
        round(r2, 4)
        if pd.notna(r2)
        else np.nan
    ]
})


st.dataframe(
    validation,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# STATE SUMMARY
# ============================================================

st.subheader(
    "Spending-state distribution"
)


state_summary = (
    out["State"]
    .value_counts()
    .rename_axis("State")
    .reset_index(name="Number of observations")
)


st.dataframe(
    state_summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# EXCEL DOWNLOAD
# ============================================================

buffer = io.BytesIO()


with pd.ExcelWriter(
    buffer,
    engine="openpyxl"
) as writer:

    out.to_excel(
        writer,
        index=False,
        sheet_name="MBRL Results"
    )


    methodology = pd.DataFrame({

        "Parameter": [

            "Threshold τ",

            "Deviation formula",

            "Classification rule",

            "World model",

            "Policy optimisation",

            "Candidate range"

        ],

        "Value": [

            tau,

            "(Actual − MBRL) / MBRL × 100",

            "D < −τ = Underspending; "
            "−τ ≤ D ≤ τ = Normal; "
            "D > τ = Overspending",

            "Random Forest regression",

            "Minimum-cost model-based policy",

            f"{min_spending}% to "
            f"{max_spending}% GDP"

        ]

    })


    methodology.to_excel(
        writer,
        index=False,
        sheet_name="Methodology"
    )


# ============================================================
# DOWNLOAD BUTTON
# ============================================================

st.download_button(
    label="Download MBRL Excel Results",
    data=buffer.getvalue(),
    file_name="MBRL_Education_Spending_Results.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    )
)
