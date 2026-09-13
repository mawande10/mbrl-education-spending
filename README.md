# MBRL Education Spending Streamlit App

Upload an Excel workbook containing **Country**, **Year**, and **Actual WB % GDP**. The app learns a transition/world model from historical education-spending observations, simulates candidate expenditure actions, selects an MBRL-derived policy-reference level, then calculates:

`MBRL-derived % GDP → Deviation → State`

Deviation: `(Actual WB % GDP - MBRL-derived % GDP) / MBRL-derived % GDP * 100`

Default classification threshold τ = 10%:
- D < -10%: Underspending
- -10% ≤ D ≤ +10%: Normal Spending
- D > +10%: Overspending

The ±10% rule is an operational research threshold, not an intrinsic MBRL rule.

## GitHub → Streamlit Cloud
1. Create a GitHub repository, e.g. `mbrl-education-spending`.
2. Upload `app.py` and `requirements.txt`.
3. In Streamlit Community Cloud choose **Deploy an app**.
4. Select the repository and `app.py` as the main file.
5. Deploy.
6. Upload the Excel workbook in the Streamlit app.
