
import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="MBRL Education Spending Assessment",
    layout="wide"
)

st.title("MBRL Education Spending Assessment")

st.write(
    "World Bank Education Spending → "
    "MBRL-derived % GDP → Deviation → Spending State"
)


# ============================================================
# MBRL PARAMETERS
# ============================================================

TAU = 0.10

st.sidebar.header("Model Parameters")

trees = st.sidebar.slider(
    "World-model trees",
    min_value=100,
    max_value=800,
    value=400,
    step=50
)

st.sidebar.write("Classification threshold: ±10%")


# ============================================================
# EXCEL UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
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
# FIND COLUMN
# ============================================================

def find_column(dataframe, possible_names):

    columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for name in possible_names:

        key = name.strip().lower()

        if key in columns:
            return columns[key]

    return None


# ============================================================
# NO FILE UPLOADED
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload an Excel file containing at least "
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
        uploaded_file
    )

except Exception as error:

    st.error(
        f"Unable to read the Excel file: {error}"
    )

    st.stop()


st.success(
    f"Excel uploaded successfully: {len(raw)} rows"
)


# ============================================================
# SHOW ORIGINAL COLUMNS
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


missing_columns = []


if country_column is None:
    missing_columns.append(
        "Country"
    )


if year_column is None:
    missing_columns.append(
        "Year"
    )


if actual_column is None:
    missing_columns.append(
        "Actual WB % GDP"
    )


if missing_columns:

    st.error(
        "Required columns not found: "
        + ", ".join(missing_columns)
    )

    st.write(
        "Your Excel contains these columns:"
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
        country_column: "Country",
        year_column: "Year",
        actual_column: "Actual_WB_GDP"
    }
).copy()


# ============================================================
# FIND OPTIONAL VARIABLES
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
# CLEAN REQUIRED VARIABLES
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


# ============================================================
# CLEAN OPTIONAL VARIABLES
# ============================================================

optional_columns = [
    "GDP",
    "GDP_Growth",
    "GDP_pc",
    "Population",
    "EduGov"
]


for column in optional_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# REMOVE INVALID YEARS
# ============================================================

df = df[
    df["Year"].notna()
].copy()


df["Year"] = (
    df["Year"]
    .astype(int)
)


# ============================================================
# SORT DATA
# ============================================================

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

number_countries = (
    df["Country"]
    .nunique()
)

number_years = (
    df["Year"]
    .nunique()
)

number_valid = (
    df["Actual_WB_GDP"]
    .notna()
    .sum()
)


st.info(
    f"Countries: {number_countries} | "
    f"Years: {number_years} | "
    f"Valid WB observations: {number_valid}"
)


# ============================================================
# CREATE LAG FEATURES
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
        lambda series:
        series
        .shift(1)
        .rolling(
            window=3,
            min_periods=1
        )
        .mean()
    )
)


# ============================================================
# NORMALISED YEAR
# ============================================================

minimum_year = df["Year"].min()

maximum_year = df["Year"].max()

year_difference = max(
    1,
    maximum_year - minimum_year
)


df["YearNorm"] = (
    (df["Year"] - minimum_year)
    / year_difference
)


# ============================================================
# OPTIONAL FEATURES
# ============================================================

extra_features = [

    column

    for column in [
        "GDP",
        "GDP_Growth",
        "GDP_pc",
        "Population",
        "EduGov"
    ]

    if column in df.columns
]


# ============================================================
# MBRL WORLD MODEL FEATURES
# ============================================================

features = [
    "Lag1",
    "Lag2",
    "Roll3",
    "YearNorm",
    "Action"
] + extra_features


# ============================================================
# BUILD TRANSITIONS
#
# State(t) + Action(t)
#          ↓
# World Model
#          ↓
# Education Spending(t+1)
# ============================================================

transition_rows = []


for country, country_data in df.groupby(
    "Country"
):

    country_data = (
        country_data
        .sort_values("Year")
        .reset_index(drop=True)
    )


    for i in range(
        len(country_data) - 1
    ):

        current = country_data.iloc[i]

        next_row = country_data.iloc[i + 1]


        if pd.isna(
            current["Actual_WB_GDP"]
        ):

            continue


        if pd.isna(
            next_row["Actual_WB_GDP"]
        ):

            continue


        row = {}


        for feature in features:

            if feature == "Action":

                row[feature] = (
                    current["Actual_WB_GDP"]
                )

            else:

                row[feature] = current.get(
                    feature,
                    np.nan
                )


        row["Target"] = (
            next_row["Actual_WB_GDP"]
        )


        transition_rows.append(
            row
        )


# ============================================================
# TRANSITION DATAFRAME
# ============================================================

transitions = pd.DataFrame(
    transition_rows
)


if len(transitions) < 5:

    st.error(
        f"Only {len(transitions)} usable "
        "transitions were created."
    )

    st.warning(
        "The dataset needs at least 5 usable "
        "year-to-year observations."
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


# ============================================================
# MEDIAN IMPUTATION
# ============================================================

feature_medians = X.median(
    numeric_only=True
)


X = X.fillna(
    feature_medians
)


X = X.fillna(0)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

split_point = int(
    len(X) * 0.80
)


split_point = max(
    1,
    min(
        split_point,
        len(X) - 1
    )
)


X_train = X.iloc[
    :split_point
]


X_test = X.iloc[
    split_point:
]


y_train = y.iloc[
    :split_point
]


y_test = y.iloc[
    split_point:
]


# ============================================================
# RANDOM FOREST WORLD MODEL
# ============================================================

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
# WORLD MODEL VALIDATION
# ============================================================

predictions = (
    world_model
    .predict(X_test)
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
#
# Possible education expenditure targets
# from 0.5% to 12% of GDP
# ============================================================

candidate_actions = np.arange(
    0.5,
    12.01,
    0.1
)


# ============================================================
# MBRL POLICY
# ============================================================

def mbrl_policy(row):

    # --------------------------------------------------------
    # Current spending reference
    # --------------------------------------------------------

    if pd.notna(
        row["Lag1"]
    ):

        current_level = (
            row["Lag1"]
        )

    elif pd.notna(
        row["Roll3"]
    ):

        current_level = (
            row["Roll3"]
        )

    elif pd.notna(
        row["Actual_WB_GDP"]
    ):

        current_level = (
            row["Actual_WB_GDP"]
        )

    else:

        current_level = (
            df["Actual_WB_GDP"]
            .median()
        )


    candidate_scores = []


    # --------------------------------------------------------
    # Test each possible action
    # --------------------------------------------------------

    for action in candidate_actions:

        state_action = {}


        for feature in features:

            if feature == "Action":

                state_action[
                    feature
                ] = action

            else:

                state_action[
                    feature
                ] = row.get(
                    feature,
                    np.nan
                )


        test_row = pd.DataFrame(
            [state_action],
            columns=features
        )


        test_row = test_row.replace(
            [np.inf, -np.inf],
            np.nan
        )


        test_row = test_row.fillna(
            feature_medians
        )


        test_row = test_row.fillna(0)


        # ----------------------------------------------------
        # Predict next education expenditure
        # ----------------------------------------------------

        predicted_next = (
            world_model
            .predict(test_row)[0]
        )


        # ----------------------------------------------------
        # MBRL objective
        # ----------------------------------------------------

        prediction_error = (
            predicted_next - action
        ) ** 2


        stability_penalty = (
            action - current_level
        ) ** 2


        # Penalise unrealistically low expenditure
        low_penalty = max(
            0,
            4 - action
        ) ** 2


        # Penalise unusually high expenditure
        high_penalty = max(
            0,
            action - 8
        ) ** 2


        total_score = (

            prediction_error

            + 0.25
            * stability_penalty

            + 0.10
            * (
                low_penalty
                + high_penalty
            )
        )


        candidate_scores.append(
            (
                total_score,
                action
            )
        )


    # --------------------------------------------------------
    # Select lowest-cost policy action
    # --------------------------------------------------------

    best = min(
        candidate_scores,
        key=lambda value: value[0]
    )


    return float(
        best[1]
    )


# ============================================================
# CALCULATE MBRL-DERIVED % GDP
# ============================================================

mbrl_results = []


for _, row in df.iterrows():

    if pd.notna(
        row["Actual_WB_GDP"]
    ):

        value = mbrl_policy(
            row
        )

    else:

        value = np.nan


    mbrl_results.append(
        value
    )


df["MBRL_Derived_GDP"] = (
    mbrl_results
)


# ============================================================
# CALCULATE DEVIATION
#
# D = ((Actual - MBRL) / MBRL) × 100
# ============================================================

df["Deviation"] = np.nan


valid_rows = (

    df["Actual_WB_GDP"].notna()

    & df["MBRL_Derived_GDP"].notna()

    & (
        df["MBRL_Derived_GDP"] != 0
    )
)


df.loc[
    valid_rows,
    "Deviation"
] = (

    (
        df.loc[
            valid_rows,
            "Actual_WB_GDP"
        ]

        -

        df.loc[
            valid_rows,
            "MBRL_Derived_GDP"
        ]
    )

    /

    df.loc[
        valid_rows,
        "MBRL_Derived_GDP"
    ]

) * 100


# ============================================================
# CLASSIFICATION
#
# FIXED 10% THRESHOLD
# ============================================================

def classify_spending(deviation):

    if pd.isna(
        deviation
    ):

        return "Not classified"


    if deviation < -10:

        return "Underspending"


    if deviation > 10:

        return "Overspending"


    return "Normal Spending"


df["State"] = (
    df["Deviation"]
    .apply(
        classify_spending
    )
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
# DISPLAY RESULTS
# ============================================================

st.subheader(
    "MBRL Results"
)


display_results = (
    results.copy()
)


display_results[
    "Actual WB % GDP"
] = (
    display_results[
        "Actual WB % GDP"
    ].round(2)
)


display_results[
    "MBRL-derived % GDP"
] = (
    display_results[
        "MBRL-derived % GDP"
    ].round(2)
)


display_results[
    "Deviation"
] = (
    display_results[
        "Deviation"
    ].apply(
        lambda value:
        f"{value:+.2f}%"
        if pd.notna(value)
        else ""
    )
)


st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CHECK CALCULATION
# ==========================================

