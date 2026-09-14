import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Streamlit UI setup ---
st.set_page_config(page_title='MBRL Education Spending', layout='wide')
st.title('MBRL Education Spending Assessment')
st.write('Excel upload → MBRL-derived % GDP → Deviation → Spending State')

tau = st.sidebar.slider('Threshold τ', 0.01, 0.30, 0.10, 0.01)
trees = st.sidebar.slider('World-model trees', 100, 800, 400, 50)

up = st.file_uploader('Upload Excel workbook', type=['xlsx'])

# --- Column aliases ---
ALIASES = {
    'Country':['Country','Country Name','geoUnit'],
    'Year':['Year','year'],
    'Actual':['Actual WB % GDP','Actual WB %GDP','WB % GDP','Education % GDP','Education expenditure % GDP','SE.XPD.TOTL.GD.ZS'],
    'GDP':['GDP','GDP (current US$)','NY.GDP.MKTP.CD'],
    'GDP_Growth':['GDP Growth','GDP growth (annual %)','NY.GDP.MKTP.KD.ZG'],
    'GDP_pc':['GDP per capita','NY.GDP.PCAP.CD'],
    'Population':['Population','SP.POP.TOTL'],
    'EduGov':['Education % Government Expenditure','Education expenditure % government expenditure','SE.XPD.TOTL.GB.ZS']
}

def col(df, names):
    m={str(c).strip().lower():c for c in df.columns}
    for n in names:
        if n.lower() in m: return m[n.lower()]
    return None

# --- Default view if no file uploaded ---
if not up:
    st.info('Required columns: Country, Year, Actual WB % GDP')
    st.dataframe(pd.DataFrame({
        'Country':['Country A','Country B','Country C'],
        'Year':[2022]*3,
        'Actual WB % GDP':[6.2,2.1,8.5]
    }), use_container_width=True)
    st.stop()

# --- Load and clean data ---
raw = pd.read_excel(up)
cc, yy, aa = col(raw, ALIASES['Country']), col(raw, ALIASES['Year']), col(raw, ALIASES['Actual'])
if not all([cc, yy, aa]):
    st.error('Excel must contain columns equivalent to Country, Year and Actual WB % GDP.')
    st.stop()

df = raw.rename(columns={cc:'Country', yy:'Year', aa:'Actual_WB_GDP'}).copy()
for k in ['GDP','GDP_Growth','GDP_pc','Population','EduGov']:
    c = col(df, ALIASES[k])
    if c and c != 'Country':
        df = df.rename(columns={c:k})

df['Country'] = df['Country'].astype(str).str.strip()
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Actual_WB_GDP'] = pd.to_numeric(df['Actual_WB_GDP'], errors='coerce')
for c in ['GDP','GDP_Growth','GDP_pc','Population','EduGov']:
    if c in df: df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.sort_values(['Country','Year']).reset_index(drop=True)

# --- Feature engineering ---
g = df.groupby('Country', group_keys=False)
df['Lag1'] = g['Actual_WB_GDP'].shift(1)
df['Lag2'] = g['Actual_WB_GDP'].shift(2)
df['Roll3'] = g['Actual_WB_GDP'].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
df['YearNorm'] = (df['Year'] - df['Year'].min()) / max(1, df['Year'].max() - df['Year'].min())

extra = [c for c in ['GDP','GDP_Growth','GDP_pc','Population','EduGov'] if c in df]
features = ['Lag1','Lag2','Roll3','YearNorm','Action'] + extra

# --- Build transition dataset ---
rows = []
for country, x in df.groupby('Country'):
    x = x.sort_values('Year').reset_index(drop=True)
    for i in range(len(x)-1):
        if pd.isna(x.loc[i,'Actual_WB_GDP']) or pd.isna(x.loc[i+1,'Actual_WB_GDP']):
            continue
        r = {f: x.loc[i,f] if f!='Action' else x.loc[i,'Actual_WB_GDP'] for f in features}
        r['Target'] = x.loc[i+1,'Actual_WB_GDP']
        rows.append(r)

trans = pd.DataFrame(rows)
if len(trans) < 5:
    st.error('At least 5 historical transitions are required. Preferably upload 54 African countries × 2015–2025.')
    st.stop()

# --- Train model ---
X = trans[features].replace([np.inf,-np.inf],np.nan)
X = X.fillna(X.median(numeric_only=True)).fillna(0)
y = trans.Target

split = max(1, int(len(X)*.8))
model = RandomForestRegressor(n_estimators=trees, random_state=42, min_samples_leaf=2, n_jobs=-1)
model.fit(X.iloc[:split], y.iloc[:split])

pred = model.predict(X.iloc[split:])
mae = mean_absolute_error(y.iloc[split:], pred)
rmse = mean_squared_error(y.iloc[split:], pred)**.5
r2 = r2_score(y.iloc[split:], pred) if len(y.iloc[split:]) > 1 else np.nan

# --- Policy search ---
cands = np.arange(2.0, 10.01, 0.1)  # tightened to realistic 2–10% range

def policy(r):
    vals = []
    lag = r.Lag1 if pd.notna(r.Lag1) else (r.Roll3 if pd.notna(r.Roll3) else r.Actual_WB_GDP)
    for a in cands:
        z = {f:(a if f=='Action' else r.get(f,np.nan)) for f in features}
        z = pd.DataFrame([z], columns=features).replace([np.inf,-np.inf],np.nan).fillna(X.median(numeric_only=True)).fillna(0)
        nxt = model.predict(z)[0]
        score = (nxt-a)**2 + .25*(a-lag)**2 + .10*(max(0,4-a)**2 + max(0,a-6)**2)
        vals.append((score,a))
    return min(vals)[1]

def calc(r):
    m = policy(r) if pd.notna(r.Actual_WB_GDP) else np.nan
    d = (r.Actual_WB_GDP - m)/m if pd.notna(m) and m!=0 and pd.notna(r.Actual_WB_GDP) else np.nan
    state = 'Underspending' if pd.notna(d) and d < -tau else 'Overspending' if pd.notna(d) and d > tau else 'Normal Spending' if pd.notna(d) else 'Not classified'
    return pd.Series([m, d*100 if pd.notna(d) else np.nan, state])

df[['MBRL_Derived_GDP','Deviation','State']] = df.apply(calc, axis=1)

# --- Results output ---
out = df[['Country','Year','Actual_WB_GDP','MBRL_Derived_GDP','Deviation','State']].copy()
out.columns = ['Country','Year','Actual WB % GDP','MBRL-derived % GDP','Deviation','State']

st.subheader('Automated Results')
show = out.copy()
show['Actual WB % GDP'] = show['Actual WB % GDP'].round(2)
show['MBRL-derived % GDP'] = show['MBRL-derived % GDP'].round(2)
show['Deviation'] = show['Deviation'].map(lambda x: f'{x:+.2f}%' if pd.notna(x) else '')
st.dataframe(show, use_container_width=True, hide_index=True)

m1,m2,m3,m4 = st.columns(4)
m1.metric('Records', len(out))
m2.metric('MBRL mean', f"{out['MBRL-derived % GDP'].mean():.2f}%")
m3.metric('Actual mean', f"{out['Actual WB % GDP'].mean():.2f}%")
m4.metric('Threshold', f'±{tau:.0%}')

st.subheader('World-model validation')
st.write({'MAE': round(mae,4), 'RMSE': round(rmse,4), 'R²': round(r2,4) if pd.notna(r2) else None})

# --- Download results ---
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine='openpyxl') as w:
    out.to_excel(w, index=False, sheet_name='MBRL Results')
    pd.DataFrame({
        'Parameter': ['τ','Deviation formula','Classification'],
        'Value': [
            tau,
            '(Actual − MBRL)/MBRL × 100',
            f'D < −{tau:.0%} Underspending; −{tau:.0%} ≤ D ≤ {tau:.0%} Normal; D > {tau:.0%} Overspending'
        ]
    }).to_excel(w, index=False, sheet_name='Methodology')

st.download_button(
    'Download MBRL Excel Results',
    buf.getvalue(),
    'MBRL_Education_Spending_Results.xlsx'
)

