# app.py - Streamlit UI for NSE LLM Bot (Hugging Face Space ready)
import streamlit as st
import pandas as pd, joblib, time
from pathlib import Path
from backend.llm_wrapper import LocalLLM
from backend.expert_rules import apply_rules
import matplotlib.pyplot as plt

st.set_page_config(page_title="NSE LLM Bot", layout="wide")
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "backend" / "models" / "rf_model.joblib"
DATA_PATH = ROOT / "sample_stocks_preview.csv"

@st.cache_data(ttl=300)
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=['timestamp'])

model = load_model()
df_all = load_data()

# Simple dark-ish CSS
st.markdown("""
<style>
body {{ background-color: #0e1117; color: #c9d1d9; }}
.stButton>button {{ background-color: #2563eb; color: white; }}
</style>
""", unsafe_allow_html=True)

tabs = st.tabs(["Overview","Predictions","LLM Chat","Expert Rules","Model Metrics"])

with tabs[0]:
    st.header("Overview")
    st.write("Nairobi Stock Exchange — LLM Bot (paper-trading prototype).")
    st.write("Select a ticker in Predictions to get local inference.")

with tabs[1]:
    st.header("Predictions")
    ticker = st.selectbox("Ticker", options=sorted(df_all['name'].unique()))
    df = df_all[df_all['name']==ticker].sort_values('timestamp').reset_index(drop=True)
    st.write(df.tail(10)[['timestamp','last','high','low','vol_']].rename(columns={'last':'close','vol_':'volume'}))
    latest = df.iloc[-1]
    prev_close = df['last'].shift(1).iloc[-1] if len(df)>1 else latest['last']
    feat = pd.DataFrame([{{ 'last': latest['last'], 'return_1': (latest['last']/prev_close - 1), 'sma_5': df['last'].rolling(5).mean().iloc[-1], 'sma_10': df['last'].rolling(10).mean().iloc[-1], 'vol_10': df['vol_'].rolling(10).mean().iloc[-1] }}]).fillna(0)
    prob = float(model.predict_proba(feat)[0,1])
    st.metric("Model probability (next-day >0.5% return)", f"{prob:.2%}")
    use_llm = st.checkbox("Use local LLM sentiment", value=True)
    force_fallback = st.checkbox("Force fallback (no HF)")
    llm = LocalLLM(use_fallback=force_fallback)
    if use_llm:
        with st.spinner("Generating local LLM sentiment..."):
            s = llm.generate_sentiment(f"{ticker} latest change {latest.get('chg_%',0)}%")
            st.metric("LLM sentiment (−1..1)", f"{s:.2f}")
    else:
        s = 0.0
    combined = 0.7*prob + 0.3*((s+1)/2)
    raw_signal = 'BUY' if prob>0.6 else ('SELL' if prob<0.4 else 'HOLD')
    row = dict(model_signal=raw_signal, return_5d=float(df['last'].pct_change(5).iloc[-1]), vol_10=float(df['vol_'].rolling(10).mean().iloc[-1]), low=float(latest.get('low',0)), high=float(latest.get('high',0)), chg_%=float(latest.get('chg_%',0)), llm_sentiment=s)
    adjusted, reasons = apply_rules(row)
    final = 'HOLD' if adjusted=='HOLD' else ('BUY (STRONG)' if adjusted=='BUY_STRONG' else ('BUY' if combined>0.6 else ('SELL' if combined<0.4 else 'HOLD')))
    st.write(f"Raw: {raw_signal} — Adjusted: {adjusted} — Final: {final}")
    if reasons: st.warning('Reasons: ' + '; '.join(reasons))
    fig, ax = plt.subplots()
    ax.plot(df['timestamp'], df['last'], label='price')
    ax.plot(df['timestamp'], df['last'].rolling(5).mean(), label='sma5')
    ax.plot(df['timestamp'], df['last'].rolling(10).mean(), label='sma10')
    ax.legend(); ax.set_title(f"{ticker} price")
    st.pyplot(fig)

with tabs[2]:
    st.header("LLM Chat (local)")
    st.write("Basic local LLM sentiment generator (not a full chat model).")
    txt = st.text_area("Enter short prompt (e.g., 'market view on KCB today')", value="market looks optimistic")
    if st.button("Get sentiment"):
        llm2 = LocalLLM(use_fallback=False)
        score = llm2.generate_sentiment(txt)
        st.success(f"Sentiment score: {score:.2f} (−1..1)")

with tabs[3]:
    st.header("Expert Rules")
    st.code(open('backend/expert_rules.py').read(), language='python')

with tabs[4]:
    st.header("Model Metrics")
    st.write("Model trained on uploaded stocks.csv")
    st.json({"accuracy": 0.9877937137625877, "auc": 0.7388288516878894})
    with open(MODEL_PATH,'rb') as fh:
        st.download_button('Download model', data=fh, file_name='rf_model.joblib')
