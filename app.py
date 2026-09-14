
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
    min_value=50,
    max_value=400,
    value=100,
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

def find_column(dataframe, possible_names):

    lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for name in possible_names:

        key = str(name).strip().lower()

        if key in lookup:
            return lookup[key]

    return None


# ============================================================
# NO FILE
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
        uploaded_file,
        engine="openpyxl"
    )

except Exception as error:

    st.error(
        f"Unable to read Excel file: {error}"
    )

    st.stop()


if raw.empty:

    st.error(
        "The uploaded Excel workbook contains no data."
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
        "Columns found in your Excel:"
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
# CLEAN REQUIRED DATA
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
# CLEAN OPTIONAL DATA
# ============================================================

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


df["Year"] = (
    df["Year"]
    .astype(int)
)


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
# BASIC DATA CHECK
# ============================================================

if number_valid < 5:

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
    (
        df["Year"]
        - minimum_year
    )
    /
    year_difference
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
# BUILD TRANSITIONS
#
# State(t) + Action(t)
#          ↓
# World Model
#          ↓
# Spending(t+1)
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

        current = (
            country_data.iloc[i]
        )

        next_row = (
            country_data.iloc[i + 1]
        )


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

                transition[
                    feature
                ] = current[
                    "Actual_WB_GDP"
                ]

            else:

                transition[
                    feature
                ] = current.get(
                    feature,
                    np.nan
                )


        transition[
            "Target"
        ] = next_row[
            "Actual_WB_GDP"
        ]


        transition_rows.append(
            transition
        )


# ============================================================
# TRANSITIONS DATAFRAME
# ============================================================

transitions = pd.DataFrame(
    transition_rows
)


if len(transitions) < 5:

    st.error(
        f"Only {len(transitions)} usable "
        "year-to-year transitions were created."
    )

    st.warning(
        "Each country should preferably contain "
        "multiple years of education-spending data."
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
# IMPUTE FEATURES
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
# Candidate education spending targets
# ============================================================

candidate_actions = np.arange(
    0.5,
    12.01,
    0.1
)


# ============================================================
# FAST MBRL POLICY
#
# IMPORTANT:
# Only ONE policy function exists in this app.
#
# Candidate actions are predicted in ONE batch
# rather than calling Random Forest 116 times.
# ============================================================

def mbrl_policy(row):

    # --------------------------------------------------------
    # Current education spending
    # --------------------------------------------------------

    if pd.notna(
        row["Actual_WB_GDP"]
    ):

        current_level = float(
            row["Actual_WB_GDP"]
        )

    elif pd.notna(
        row["Lag1"]
    ):

        current_level = float(
            row["Lag1"]
        )

    elif pd.notna(
        row["Roll3"]
    ):

        current_level = float(
            row["Roll3"]
        )

    else:

        current_level = float(
            df["Actual_WB_GDP"]
            .median()
        )


    # --------------------------------------------------------
    # Historical trajectory
    # --------------------------------------------------------

    if pd.notna(
        row["Roll3"]
    ):

        historical_target = float(
            row["Roll3"]
        )

    else:

        historical_target = current_level


    # --------------------------------------------------------
    # CREATE ALL ACTIONS AT ONCE
    # --------------------------------------------------------

    candidate_data = pd.DataFrame(
        {
            feature: [

                (
                    action

                    if feature == "Action"

                    else row.get(
                        feature,
                        np.nan
                    )
                )

                for action
                in candidate_actions

            ]

            for feature
            in features
        }
    )


    # --------------------------------------------------------
    # CLEAN CANDIDATE DATA
    # --------------------------------------------------------

    candidate_data = (
        candidate_data
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )


    candidate_data = (
        candidate_data
        .fillna(
            feature_medians
        )
    )


    candidate_data = (
        candidate_data
        .fillna(0)
    )


    # --------------------------------------------------------
    # PREDICT ALL CANDIDATE ACTIONS
    # IN ONE RANDOM FOREST CALL
    # --------------------------------------------------------

    predicted_next = (
        world_model
        .predict(
            candidate_data
        )
    )


    # --------------------------------------------------------
    # OBJECTIVE 1
    #
    # Model consistency
    # --------------------------------------------------------

    prediction_error = (
        predicted_next
        - candidate_actions
    ) ** 2


    # --------------------------------------------------------
    # OBJECTIVE 2
    #
    # Avoid unnecessary spending jumps
    # --------------------------------------------------------

    stability_penalty = (
        candidate_actions
        - current_level
    ) ** 2


    # --------------------------------------------------------
    # OBJECTIVE 3
    #
    # Stay close to historical trajectory
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
    # SELECT OPTIMAL ACTION
    # --------------------------------------------------------

    best_index = int(
        np.argmin(
            total_cost
        )
    )


    best_action = float(
        candidate_actions[
            best_index
        ]
    )


    return best_action


# ============================================================
# RUN MBRL
# ============================================================

st.subheader(
    "MBRL Policy Optimisation"
)


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


    df[
        "MBRL_Derived_GDP"
    ] = mbrl_values


# ============================================================
# MBRL CALCULATION CHECK
# ============================================================

number_mbrl = int(
    df[
        "MBRL_Derived_GDP"
    ]
    .notna()
    .sum()
)


st.success(
    f"MBRL optimisation completed: "
    f"{number_mbrl} MBRL values calculated."
)


# ============================================================
# DEVIATION
#
# D = ((Actual - MBRL) / MBRL) × 100
# ============================================================

df["Deviation"] = np.nan


valid_rows = (

    df[
        "Actual_WB_GDP"
    ].notna()

    &

    df[
        "MBRL_Derived_GDP"
    ].notna()

    &

    (
        df[
            "MBRL_Derived_GDP"
        ] != 0
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
# D < -10%       = Underspending
# -10% to +10%   = Normal Spending
# D > +10%       = Overspending
# ============================================================

def classify_spending(
    deviation
):

    if pd.isna(
        deviation
    ):

        return "Not classified"


    if deviation < -10:

        return "Underspending"


    if deviation > 10:

        return "Overspending"


    return "Normal Spending"


df[
    "State"
] = (
    df[
        "Deviation"
    ]
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
# CALCULATION CHECK
# ============================================================

st.subheader(
    "Calculation Check"
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Total rows",
    len(results)
)


col2.metric(
    "MBRL calculated",
    int(
        results[
            "MBRL-derived % GDP"
        ]
        .notna()
        .sum()
    )
)


col3.metric(
    "Deviation calculated",
    int(
        results[
            "Deviation"
        ]
        .notna()
        .sum()
    )
)


col4.metric(
    "Classified",
    int(
        (
            results[
                "State"
            ]
            != "Not classified"
        )
        .sum()
    )
)


# ============================================================
# RESULTS TABLE
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
    ]
    .round(2)
)


display_results[
    "MBRL-derived % GDP"
] = (
    display_results[
        "MBRL-derived % GDP"
    ]
    .round(2)
)


display_results[
    "Deviation"
] = (
    display_results[
        "Deviation"
    ]
    .apply(
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
# STATE SUMMARY
# ============================================================

st.subheader(
    "Spending State Summary"
)


state_summary = (
    results[
        "State"
    ]
    .value_counts()
    .rename_axis("State")
    .reset_index(
        name="Number of observations"
    )
)


st.dataframe(
    state_summary,
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
# METHODOLOGY
# ============================================================

st.subheader(
    "Methodology"
)


st.markdown(
    """
### MBRL workflow

**State representation**

Historical education expenditure, lagged expenditure,
rolling expenditure and available economic variables.

**Action**

Candidate education expenditure target:

\[
A_{c,t} \in [0.5,12.0]\% \ GDP
\]

**World model**

The Random Forest world model estimates:

\[
\hat{E}_{c,t+1}
=
f_\theta(S_{c,t},A_{c,t})
\]

**Policy**

The MBRL policy searches candidate actions and selects
the action with the lowest model-based policy cost.

**Deviation**

\[
D_{c,t}
=
\frac{
E^{Actual}_{c,t}
-
E^{MBRL}_{c,t}
}{
E^{MBRL}_{c,t}
}
\times100
\]

**Operational classification rule**

- \(D < -10\%\): **Underspending**
- \(-10\% \leq D \leq +10\%\): **Normal Spending**
- \(D > +10\%\): **Overspending**
"""
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


    state_summary.to_excel(
        writer,
        index=False,
        sheet_name="State Summary"
    )


    methodology = pd.DataFrame({

        "Parameter": [

            "World Bank indicator",

            "MBRL-derived % GDP",

            "Deviation formula",

            "Classification threshold",

            "Underspending",

            "Normal Spending",

            "Overspending"

        ],

        "Definition": [

            "SE.XPD.TOTL.GD.ZS",

            "Model-based policy-selected education spending target",

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

    file_name=(
        "MBRL_Education_Spending_Results.xlsx"
    ),

    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    )
)


# ============================================================
# END
# ============================================================


