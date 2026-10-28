import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "churn_stacked_lr_model.pkl")
scaler = joblib.load(BASE_DIR / "churn_scale.pkl")

FEATURE_COLS = [
    'tenure', 'TotalCharges', 'MonthlyCharges',
    'InternetService_Fiber optic', 'Contract_Two year', 'PaymentMethod_Electronic check',
    'DeviceProtection_No internet service', 'TechSupport_No internet service',
    'StreamingMovies_No internet service', 'OnlineBackup_No internet service',
    'OnlineSecurity_No internet service', 'InternetService_No',
    'StreamingTV_No internet service', 'PaperlessBilling_Yes', 'Contract_One year',
    'OnlineSecurity_Yes', 'TechSupport_Yes', 'Dependents_Yes', 'Partner_Yes',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Mailed check',
    'OnlineBackup_Yes', 'DeviceProtection_Yes', 'StreamingTV_Yes',
    'StreamingMovies_Yes', 'MultipleLines_Yes',
]

SERVICES = {
    'OnlineSecurity': 'security', 'OnlineBackup': 'backup',
    'DeviceProtection': 'protection', 'TechSupport': 'tech',
    'StreamingTV': 'tv', 'StreamingMovies': 'movies',
}

ACTIONS = {
    'contract': 'offer a 12-month contract discount',
    'tenure': 'run an onboarding / loyalty programme',
    'monthly': 'review the plan price or offer a bundle discount',
    'internet': 'check fiber service quality and pricing',
    'payment': 'incentivise auto-pay (small bill credit)',
    'paperless': 'revisit billing preferences',
    'partner': 'target with a family / multi-line plan',
    'dependents': 'target with a family / multi-line plan',
    'tech': 'include free tech support',
    'security': 'offer a free online security trial',
}


def build_row(p):
    row = {
        'tenure': p['tenure'],
        'TotalCharges': p['total'],
        'MonthlyCharges': p['monthly'],
        'Partner_Yes': p['partner'] == 'Yes',
        'Dependents_Yes': p['dependents'] == 'Yes',
        'PaperlessBilling_Yes': p['paperless'] == 'Yes',
        'MultipleLines_Yes': p['multiple'] == 'Yes',
        'InternetService_Fiber optic': p['internet'] == 'Fiber optic',
        'InternetService_No': p['internet'] == 'No',
        'Contract_One year': p['contract'] == 'One year',
        'Contract_Two year': p['contract'] == 'Two year',
        'PaymentMethod_Electronic check': p['payment'] == 'Electronic check',
        'PaymentMethod_Credit card (automatic)': p['payment'] == 'Credit card (automatic)',
        'PaymentMethod_Mailed check': p['payment'] == 'Mailed check',
    }
    for col, key in SERVICES.items():
        row[f'{col}_Yes'] = p[key] == 'Yes'
        row[f'{col}_No internet service'] = p[key] == 'No internet service'
    return pd.DataFrame([row])[FEATURE_COLS].astype(float)


def churn_proba(p):
    scaled = scaler.transform(build_row(p).values)
    return float(model.predict_proba(scaled)[0][1])


def explain(p, base_prob):
    flip = lambda v: 'No' if v == 'Yes' else 'Yes'
    autopay = p['payment'] in ('Credit card (automatic)', 'Bank transfer (automatic)')

    factors = [
        ('contract', f"{p['contract']} contract",
         {'contract': 'Month-to-month' if p['contract'] == 'Two year' else 'Two year'}),
        ('tenure', 'Short tenure' if p['tenure'] < 30 else 'Long tenure',
         {'tenure': 30}),
        ('monthly', 'High monthly charges' if p['monthly'] >= 70 else 'Low monthly charges',
         {'monthly': 70.0}),
        ('internet', 'No internet' if p['internet'] == 'No' else f"{p['internet']} internet",
         {'internet': 'DSL' if p['internet'] != 'DSL' else 'Fiber optic'}),
        ('payment', 'Auto-pay on' if autopay else p['payment'],
         {'payment': 'Electronic check' if autopay else 'Credit card (automatic)'}),
        ('paperless', 'Paperless billing' if p['paperless'] == 'Yes' else 'Paper billing',
         {'paperless': flip(p['paperless'])}),
        ('partner', 'Has partner' if p['partner'] == 'Yes' else 'No partner',
         {'partner': flip(p['partner'])}),
        ('dependents', 'Has dependents' if p['dependents'] == 'Yes' else 'No dependents',
         {'dependents': flip(p['dependents'])}),
    ]
    if p['internet'] != 'No':
        for key, name in [('tech', 'tech support'), ('security', 'online security')]:
            has = p[key] == 'Yes'
            factors.append((key, f"Has {name}" if has else f"No {name}",
                            {key: flip(p[key]) if p[key] in ('Yes', 'No') else 'Yes'}))

    out = []
    for key, label, change in factors:
        delta = base_prob - churn_proba({**p, **change})
        out.append({'key': key, 'label': label, 'delta': delta})
    out.sort(key=lambda d: abs(d['delta']), reverse=True)
    return out[:4]


def suggested_action(drivers, prob):
    pos = [d for d in drivers if d['delta'] > 0.01]
    if prob < 0.4 or not pos:
        return 'Customer looks stable. Keep nurturing, no urgent retention action needed.'
    acts = []
    for d in pos[:2]:
        if ACTIONS[d['key']] not in acts:
            acts.append(ACTIONS[d['key']])
    return 'Suggested action: ' + ' and '.join(acts) + '.'


st.set_page_config(page_title='Telco Churn Predictor', layout='centered')

st.markdown("""
<style>
.stApp label, .stApp label p, .stApp [data-testid="stWidgetLabel"] { color: #6b6b6b !important; }
[data-testid="stExpander"] details, [data-testid="stExpander"] details > summary {
    background: #ffffff !important; border-radius: 10px; border-color: #d9d4c9 !important;
}
[data-testid="stExpander"] details > summary *, [data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p, [data-testid="stExpander"] summary div {
    color: #111 !important; fill: #111 !important; font-weight: 600; opacity: 1 !important;
}
[data-testid="stExpander"] svg { color: #111 !important; fill: #111 !important; }
[data-testid="stExpander"] details > summary:hover { background: #f7f5f0 !important; }
.stApp { background: #f0ece4; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 620px; padding-top: 2rem; padding-bottom: 3rem; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff; border: 1px solid #e3dfd6 !important;
    border-radius: 16px; padding: 0.4rem 0.6rem;
}
.card-title { font-size: 1.05rem; font-weight: 700; color: #111; margin-bottom: 4px; }
.card-sub { font-size: 0.85rem; color: #6b6b6b; margin-bottom: 8px; }
.hdr { display:flex; align-items:center; gap:14px; margin-bottom: 14px; }
.hdr-icon { width:44px; height:44px; border-radius:12px; background:#f8d9d9;
            display:flex; align-items:center; justify-content:center; font-size:20px; }
.hdr-title { font-size:1.25rem; font-weight:700; color:#111; line-height:1.2; }
.hdr-sub { font-size:0.9rem; color:#555; }
.badge { padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
.row-head { display:flex; justify-content:space-between; align-items:center; }
.gauge-wrap { display:flex; justify-content:center; }
.drv { display:flex; align-items:center; gap:12px; margin: 7px 0; font-size:0.92rem; color:#222; }
.drv-label { width:150px; }
.drv-track { flex:1; height:10px; background:#f0ece4; border-radius:6px; overflow:hidden; }
.drv-bar { height:100%; border-radius:6px; }
.drv-val { width:48px; text-align:right; font-weight:600; }
.action { background:#fbead6; color:#6b3d0a; border-radius:10px; padding:12px 14px;
          font-size:0.92rem; margin-top:12px; }
div.stButton > button {
    width:100%; border-radius:10px; border:1px solid #cfcac0; background:#fff;
    color:#111; padding:0.6rem; font-weight:600;
}
div.stButton > button:hover { border-color:#d63c3c; color:#d63c3c; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hdr"><div class="hdr-icon">🎯</div><div>'
    '<div class="hdr-title">Telco churn predictor</div>'
    '<div class="hdr-sub">Score a customer, then see why</div></div></div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<div class="card-title">Customer profile</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    contract = c1.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    internet = c2.selectbox('Internet service', ['Fiber optic', 'DSL', 'No'])

    c1, c2 = st.columns(2)
    payment = c1.selectbox('Payment method', [
        'Electronic check', 'Credit card (automatic)', 'Mailed check', 'Bank transfer (automatic)'])
    tech = c2.selectbox('Tech support', ['No', 'Yes', 'No internet service'])

    tenure = st.slider('Tenure (months)', 0, 72, 8)
    monthly = st.slider('Monthly charges ($)', 0, 150, 89)
    total = st.number_input('Total charges ($)', min_value=0.0, value=float(tenure * monthly), step=10.0)

    with st.expander('More details'):
        c1, c2 = st.columns(2)
        partner = c1.selectbox('Partner', ['No', 'Yes'])
        dependents = c2.selectbox('Dependents', ['No', 'Yes'])
        paperless = c1.selectbox('Paperless billing', ['Yes', 'No'])
        multiple = c2.selectbox('Multiple lines', ['No', 'Yes'])
        security = c1.selectbox('Online security', ['No', 'Yes', 'No internet service'])
        backup = c2.selectbox('Online backup', ['No', 'Yes', 'No internet service'])
        protection = c1.selectbox('Device protection', ['No', 'Yes', 'No internet service'])
        tv = c2.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
        movies = c1.selectbox('Streaming movies', ['No', 'Yes', 'No internet service'])

    predict = st.button('Predict churn')

params = dict(
    tenure=tenure, monthly=float(monthly), total=total, partner=partner,
    dependents=dependents, paperless=paperless, multiple=multiple,
    internet=internet, security=security, backup=backup, protection=protection,
    tech=tech, tv=tv, movies=movies, contract=contract, payment=payment,
)

if predict:
    prob = churn_proba(params)
    st.session_state['result'] = {'prob': prob, 'drivers': explain(params, prob)}

with st.container(border=True):
    res = st.session_state.get('result')
    if res is None:
        st.markdown('<div class="card-title">Churn risk</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Fill in the profile and press “Predict churn”.</div>',
                    unsafe_allow_html=True)
    else:
        prob = res['prob']
        pct = prob * 100
        if prob >= 0.6:
            color, bg, fg, tag = '#d63c3c', '#f8d9d9', '#a52a2a', 'High risk'
        elif prob >= 0.4:
            color, bg, fg, tag = '#e29a2d', '#fbead6', '#8a5a0a', 'Medium risk'
        else:
            color, bg, fg, tag = '#2f9e5b', '#d9f1e2', '#1d6b3c', 'Low risk'

        st.markdown(
            f'<div class="row-head"><div class="card-title">Churn risk</div>'
            f'<span class="badge" style="background:{bg};color:{fg}">{tag}</span></div>',
            unsafe_allow_html=True,
        )

        gauge = (
            '<div class="gauge-wrap"><svg viewBox="0 0 240 150" width="100%" style="max-width:340px">'
            '<path d="M 20 125 A 100 100 0 0 1 220 125" pathLength="100" fill="none" '
            'stroke="#e0dedb" stroke-width="18" stroke-linecap="round"/>'
            f'<path d="M 20 125 A 100 100 0 0 1 220 125" pathLength="100" fill="none" '
            f'stroke="{color}" stroke-width="18" stroke-linecap="round" '
            f'stroke-dasharray="{max(pct, 0.5):.1f} 100"/>'
            f'<text x="120" y="100" text-anchor="middle" font-size="40" font-weight="700" fill="#111">{pct:.0f}%</text>'
            '<text x="120" y="122" text-anchor="middle" font-size="13" fill="#666">probability of churn</text>'
            '<text x="20" y="145" text-anchor="middle" font-size="11" fill="#888">0</text>'
            '<text x="220" y="145" text-anchor="middle" font-size="11" fill="#888">100</text>'
            '</svg></div>'
        )
        st.markdown(gauge, unsafe_allow_html=True)

        st.markdown('<div class="card-sub" style="margin-top:6px">What pushes this customer to leave</div>',
                    unsafe_allow_html=True)
        drivers = res['drivers']
        max_abs = max(abs(d['delta']) for d in drivers) or 1
        rows = ''
        for d in drivers:
            w = abs(d['delta']) / max_abs * 100
            bar_color = '#e04b4b' if d['delta'] > 0 else '#2f78d6'
            rows += (
                f'<div class="drv"><div class="drv-label">{d["label"]}</div>'
                f'<div class="drv-track"><div class="drv-bar" style="width:{w:.0f}%;background:{bar_color}"></div></div>'
                f'<div class="drv-val">{d["delta"]:+.2f}</div></div>'
            )
        st.markdown(rows, unsafe_allow_html=True)
        st.markdown(f'<div class="action">{suggested_action(drivers, prob)}</div>',
                    unsafe_allow_html=True)
