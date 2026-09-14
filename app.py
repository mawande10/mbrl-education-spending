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
    "World Bank Education Spending → "
    "MBRL World Model → Policy Optimisation → "
    "MBRL-derived % GDP → Deviation → Spending State"
)


# ============================================================
# PARAMETERS
# ============================================================

TAU = 10.0

st.sidebar.header("MBRL Parameters")

trees = st.sidebar.slider(
    "World-model trees",
    100,
    800,
    400,
    50
)

st.sidebar.metric(
    "Classification threshold",
    "±10%"
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Excel dataset",
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
        "geoUnit",
        "Country_Name"
    ],

    "Year": [
        "Year",
        "year",
        "YEAR"
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
# FIND COLUMN
# ============================================================

def find_column(dataframe, names):

    lookup = {
        str(c).strip().lower(): c
        for c in dataframe.columns
    }

    for name in names:

        key = str(name).strip().lower()

        if key in lookup:
            return lookup[key]

    return None


# ============================================================
# WAIT FOR FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload an Excel file containing "
        "Country, Year and Actual WB % GDP."
    )

    example = pd.DataFrame({
        "Country": [
            "South Africa",
            "South Africa",
            "South Africa",
            "Egypt",
            "Egypt",
            "Nigeria"
        ],

        "Year": [
            2020,
            2021,
            2022,
            2020,
            2021,
            2020
        ],

        "Actual WB % GDP": [
            6.20,
            5.90,
            5.50,
            4.80,
            4.50,
            1.70
        ]
    })

    st.subheader("Required Excel format")

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

    raw = pd.read_excel(
        uploaded_file,
        engine="openpyxl"
    )

except Exception as error:

    st.error(
        f"Excel could not be read: {error}"
    )

    st.stop()


if raw.empty:

    st.error(
        "The Excel workbook contains no data."
    )

    st.stop()


st.success(
    f"Excel uploaded successfully — {len(raw)} rows."
)


# ============================================================
# ORIGINAL COLUMNS
# ============================================================

with st.expander("View uploaded Excel columns"):

    st.write(
        list(raw.columns)
    )


# ============================================================
# FIND REQUIRED COLUMNS
# ============================================================

country_column = find_column(
    raw,
    ALIASES["Country"]
)

year_column = find_column(
    raw,
    ALIASES["Year"]
)

actual_column = find_column(
    raw,
    ALIASES["Actual"]
)


missing = []

if country_column is None:
    missing.append("Country")

if year_column is None:
    missing.append("Year")

if actual_column is None:
    missing.append("Actual WB % GDP")


if missing:

    st.error(
        "The following required columns were not found: "
        + ", ".join(missing)
    )

    st.write(
        "Columns detected in your Excel:"
    )

    st.write(
        list(raw.columns)
    )

    st.stop()


# ============================================================
# STANDARDISE
# ============================================================

df = raw.rename(
    columns={
        country_column: "Country",
        year_column: "Year",
        actual_column: "Actual_WB_GDP"
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

    column = find_column(
        df,
        ALIASES[variable]
    )

    if column is not None:

        if column != variable:

            df.rename(
                columns={
                    column: variable
                },
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


for column in [
    "GDP",
    "GDP_Growth",
    "GDP_pc",
    "Population",
    "EduGov"
]:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

df = df[
    df["Country"].notna()
].copy()


df = df[
    df["Year"].notna()
].copy()


df["Year"] = df["Year"].astype(int)


df = (
    df
    .sort_values(
        ["Country", "Year"]
    )
    .reset_index(drop=True)
)


# ============================================================
# DATA SUMMARY
# ============================================================

n_countries = df["Country"].nunique()

n_years = df["Year"].nunique()

n_valid = df["Actual_WB_GDP"].notna().sum()

st.info(
    f"Countries: {n_countries} | "
    f"Years: {n_years} | "
    f"Valid WB observations: {n_valid}"
)


# ============================================================
# CHECK DATA
# ============================================================

if n_valid < 5:

    st.error(
        "At least 5 valid Actual WB % GDP observations "
        "are required."
    )

    st.dataframe(
        df[
            [
                "Country",
                "Year",
                "Actual_WB_GDP"
            ]
        ],
        use_container_width=True
    )

    st.stop()


# ============================================================
# LAG FEATURES
# ============================================================

grouped = df.groupby(
    "Country",
    group_keys=False
)


df["Lag1"] = (
    grouped["Actual_WB_GDP"]
    .shift(1)
)


df["Lag2"] = (
    grouped["Actual_WB_GDP"]
    .shift(2)
)


df["Roll3"] = (
    grouped["Actual_WB_GDP"]
    .transform(
        lambda s:
        s.shift(1)
        .rolling(
            3,
            min_periods=1
        )
        .mean()
    )
)


# ============================================================
# YEAR FEATURE
# ============================================================

min_year = df["Year"].min()

max_year = df["Year"].max()

year_range = max(
    1,
    max_year - min_year
)


df["YearNorm"] = (
    (df["Year"] - min_year)
    / year_range
)


# ============================================================
# OPTIONAL FEATURES
# ============================================================

extra_features = []

for column in [
    "GDP",
    "GDP_Growth",
    "GDP_pc",
    "Population",
    "EduGov"
]:

    if column in df.columns:

        extra_features.append(
            column
        )


# ============================================================
# WORLD MODEL
# ============================================================

features = [
    "Lag1",
    "Lag2",
    "Roll3",
    "YearNorm",
    "Action"
] + extra_features


# ============================================================
# CREATE TRANSITIONS
#
# STATE(t) + ACTION(t)
#          ↓
# WORLD MODEL
#          ↓
# SPENDING(t+1)
# ============================================================

transition_rows = []


for country, data in df.groupby(
    "Country"
):

    data = (
        data
        .sort_values("Year")
        .reset_index(drop=True)
    )

    for i in range(
        len(data) - 1
    ):

        current = data.iloc[i]

        next_row = data.iloc[i + 1]

        if pd.isna(
            current["Actual_WB_GDP"]
        ):
            continue

        if pd.isna(
            next_row["Actual_WB_GDP"]
        ):
            continue

        transition = {}

        for feature in features:

            if feature == "Action":

                transition[feature] = (
                    current["Actual_WB_GDP"]
                )

            else:

                transition[feature] = (
                    current[feature]
                    if feature in current.index
                    else np.nan
                )

        transition["Target"] = (
            next_row["Actual_WB_GDP"]
        )

        transition_rows.append(
            transition
        )


transitions = pd.DataFrame(
    transition_rows
)


# ============================================================
# TRANSITION CHECK
# ============================================================

if len(transitions) < 5:

    st.error(
        f"Only {len(transitions)} usable "
        "year-to-year transitions were found."
    )

    st.warning(
        "The file needs multiple years per country."
    )

    st.stop()


st.success(
    f"{len(transitions)} transitions created "
    "for the MBRL world model."
)


# ============================================================
# MODEL DATA
# ============================================================

X = transitions[
    features
].copy()


y = transitions[
    "Target"
].copy()


X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


feature_medians = X.median()


X = X.fillna(
    feature_medians
)


X = X.fillna(0)


# ============================================================
# TRAIN WORLD MODEL
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


world_model = RandomForestRegressor(
    n_estimators=trees,
    random_state=42,
    min_samples_leaf=1,
    n_jobs=-1
)


world_model.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION
# ============================================================

predictions = world_model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    predictions
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


if len(y_test) > 1:

    r2 = r2_score(
        y_test,
        predictions
    )

else:

    r2 = np.nan


# ============================================================
# ACTION SPACE
# ============================================================

candidate_actions = np.arange(
    0.5,
    12.01,
    0.1
)


# ============================================================
# FAST MBRL POLICY
# ============================================================

candidate_actions = np.arange(
    0.5,
    12.01,
    0.1
)


def mbrl_policy_fast(row):

    # --------------------------------------------------------
    # Current spending
    # --------------------------------------------------------

    if pd.notna(row["Actual_WB_GDP"]):

        current_level = float(
            row["Actual_WB_GDP"]
        )

    elif pd.notna(row["Lag1"]):

        current_level = float(
            row["Lag1"]
        )

    elif pd.notna(row["Roll3"]):

        current_level = float(
            row["Roll3"]
        )

    else:

        current_level = float(
            df["Actual_WB_GDP"].median()
        )


    # --------------------------------------------------------
    # Historical target
    # --------------------------------------------------------

    if pd.notna(row["Roll3"]):

        historical_target = float(
            row["Roll3"]
        )

    else:

        historical_target = current_level


    # --------------------------------------------------------
    # CREATE ALL CANDIDATE ACTIONS AT ONCE
    # --------------------------------------------------------

    candidate_data = pd.DataFrame(
        {
            feature:
            [
                (
                    action
                    if feature == "Action"
                    else row.get(
                        feature,
                        np.nan
                    )
                )

                for action in candidate_actions
            ]

            for feature in features
        }
    )


    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    candidate_data = candidate_data.replace(
        [np.inf, -np.inf],
        np.nan
    )


    candidate_data = candidate_data.fillna(
        feature_medians
    )


    candidate_data = candidate_data.fillna(
        0
    )


    # --------------------------------------------------------
    # PREDICT ALL ACTIONS IN ONE CALL
    # --------------------------------------------------------

    predicted_next = world_model.predict(
        candidate_data
    )


    # --------------------------------------------------------
    # MODEL CONSISTENCY
    # --------------------------------------------------------

    prediction_error = (
        predicted_next
        - candidate_actions
    ) ** 2


    # --------------------------------------------------------
    # STABILITY PENALTY
    # --------------------------------------------------------

    stability_penalty = (
        candidate_actions
        - current_level
    ) ** 2


    # --------------------------------------------------------
    # HISTORICAL TRAJECTORY
    # --------------------------------------------------------

    trajectory_penalty = (
        candidate_actions
        - historical_target
    ) ** 2


    # --------------------------------------------------------
    # TOTAL POLICY COST
    # --------------------------------------------------------

    total_cost = (

        prediction_error

        + 0.20
        * stability_penalty

        + 0.20
        * trajectory_penalty
    )


    # --------------------------------------------------------
    # BEST ACTION
    # --------------------------------------------------------

    best_index = np.argmin(
        total_cost
    )


    return float(
        candidate_actions[
            best_index
        ]
    )


# ============================================================
# RUN MBRL OPTIMISATION
# ============================================================

with st.spinner(
    "Running fast MBRL policy optimisation..."
):

    mbrl_values = []

    for _, row in df.iterrows():

        if pd.notna(
            row["Actual_WB_GDP"]
        ):

            value = mbrl_policy_fast(
                row
            )

        else:

            value = np.nan

        mbrl_values.append(
            value
        )


df["MBRL_Derived_GDP"] = (
    mbrl_values
)


# ============================================================
# CHECK THAT MBRL CALCULATED
# ============================================================

number_mbrl = int(
    df["MBRL_Derived_GDP"]
    .notna()
    .sum()
)


st.success(
    f"MBRL optimisation completed: "
    f"{number_mbrl} MBRL values calculated."
)

# ============================================================
# CALCULATE MBRL TARGET
# ============================================================

with st.spinner(
    "Running MBRL policy optimisation..."
):

    mbrl_values = []

    for _, row in df.iterrows():

        if pd.notna(
            row["Actual_WB_GDP"]
        ):

            value = mbrl_policy(
                row
            )

        else:

            value = np.nan

        mbrl_values.append(
            value
        )


df["MBRL_Derived_GDP"] = (
    mbrl_values
)


# ============================================================
# DEVIATION
#
# D = ((Actual - MBRL) / MBRL) × 100
# ============================================================

df["Deviation"] = np.nan


valid = (

    df["Actual_WB_GDP"].notna()

    &

    df["MBRL_Derived_GDP"].notna()

    &

    (df["MBRL_Derived_GDP"] != 0)
)


df.loc[
    valid,
    "Deviation"
] = (

    (
        df.loc[
            valid,
            "Actual_WB_GDP"
        ]

        -

        df.loc[
            valid,
            "MBRL_Derived_GDP"
        ]
    )

    /

    df.loc[
        valid,
        "MBRL_Derived_GDP"
    ]

) * 100


# ============================================================
# CLASSIFICATION
# ============================================================

def classify(value):

    if pd.isna(value):

        return "Not classified"


    if value < -TAU:

        return "Underspending"


    elif value > TAU:

        return "Overspending"


    else:

        return "Normal Spending"


df["State"] = (
    df["Deviation"]
    .apply(classify)
)


# ============================================================
# FINAL RESULTS
# ============================================================

results = df[
    [
        "Country",
        "Year",
        "Actual_WB_GDP",
        "MBRL_Derived_GDP",
        "Deviation",
        "State"
    ]
].copy()


results.columns = [
    "Country",
    "Year",
    "Actual WB % GDP",
    "MBRL-derived % GDP",
    "Deviation",
    "State"
]


# ============================================================
# CALCULATION CHECK
# ============================================================

st.subheader(
    "Calculation Check"
)


c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "Rows",
    len(results)
)


c2.metric(
    "MBRL calculated",
    int(
        results[
            "MBRL-derived % GDP"
        ].notna().sum()
    )
)


c3.metric(
    "Deviation calculated",
    int(
        results[
            "Deviation"
        ].notna().sum()
    )
)


c4.metric(
    "Classified",
    int(
        (
            results["State"]
            != "Not classified"
        ).sum()
    )
)


# ============================================================
# RESULTS TABLE
# ============================================================

st.subheader(
    "MBRL Results"
)


display_results = results.copy()


display_results[
    "Actual WB % GDP"
] = display_results[
    "Actual WB % GDP"
].round(2)


display_results[
    "MBRL-derived % GDP"
] = display_results[
    "MBRL-derived % GDP"
].round(2)


display_results[
    "Deviation"
] = display_results[
    "Deviation"
].apply(
    lambda x:
    f"{x:+.2f}%"
    if pd.notna(x)
    else ""
)


st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# STATE SUMMARY
# ============================================================

st.subheader(
    "Spending State Summary"
)


state_counts = (
    results["State"]
    .value_counts()
    .rename_axis("State")
    .reset_index(name="Number of observations")
)


st.dataframe(
    state_counts,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# WORLD MODEL VALIDATION
# ============================================================

st.subheader(
    "World Model Validation"
)


v1, v2, v3 = st.columns(3)


v1.metric(
    "MAE",
    f"{mae:.4f}"
)


v2.metric(
    "RMSE",
    f"{rmse:.4f}"
)


if pd.notna(r2):

    v3.metric(
        "R²",
        f"{r2:.4f}"
    )

else:

    v3.metric(
        "R²",
        "N/A"
    )


# ============================================================
# DOWNLOAD EXCEL
# ============================================================

st.subheader(
    "Download Results"
)


buffer = io.BytesIO()


with pd.ExcelWriter(
    buffer,
    engine="openpyxl"
) as writer:

    results.to_excel(
        writer,
        index=False,
        sheet_name="MBRL Results"
    )


    methodology = pd.DataFrame({

        "Parameter": [

            "World Bank indicator",

            "MBRL-derived value",

            "Deviation",

            "Threshold",

            "Underspending",

            "Normal Spending",

            "Overspending"

        ],

        "Definition": [

            "SE.XPD.TOTL.GD.ZS",

            "Policy-selected education expenditure target as % of GDP",

            "((Actual - MBRL) / MBRL) × 100",

            "±10%",

            "Deviation < -10%",

            "-10% ≤ Deviation ≤ +10%",

            "Deviation > +10%"

        ]

    })


    methodology.to_excel(
        writer,
        index=False,
        sheet_name="Methodology"
    )


st.download_button(
    label="Download MBRL Results Excel",
    data=buffer.getvalue(),
    file_name="MBRL_Education_Spending_Results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
