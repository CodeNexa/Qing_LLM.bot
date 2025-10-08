# backend/retrain_model.py
import pandas as pd, joblib, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "stocks.csv"
MODEL_PATH = ROOT / "backend" / "models" / "rf_model.joblib"
META_PATH = ROOT / "metadata.json"

df = pd.read_csv(DATA, low_memory=False)
df = df.rename(columns=lambda c: c.strip().lower())
df['timestamp'] = pd.to_datetime(df.get('timestamp') or df.get('date'), errors='coerce')
df = df.dropna(subset=['timestamp']).sort_values(['name','timestamp'])
df['last'] = pd.to_numeric(df.get('last', df.get('close')), errors='coerce')
df['vol_'] = pd.to_numeric(df.get('vol_',0), errors='coerce').fillna(0)
df = df.dropna(subset=['last'])

def add_feats(g):
    g['return_1'] = g['last'].pct_change(1)
    g['sma_5'] = g['last'].rolling(5).mean()
    g['sma_10'] = g['last'].rolling(10).mean()
    g['vol_10'] = g['vol_'].rolling(10).mean()
    g['next_return'] = g['last'].pct_change(-1)
    g['target'] = (g['next_return'] > 0.005).astype(int)
    return g

df = df.groupby('name', group_keys=False).apply(add_feats).dropna(subset=['sma_10','target'])
X = df[['last','return_1','sma_5','sma_10','vol_10']].fillna(0)
y = df['target']

split_date = df['timestamp'].quantile(0.8)
train_mask = df['timestamp'] <= split_date
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

model = RandomForestClassifier(n_estimators=80, max_depth=7, random_state=42)
model.fit(X_train, y_train)
pred = model.predict_proba(X_test)[:,1]
acc = accuracy_score(y_test, (pred>0.5))
auc = roc_auc_score(y_test, pred) if len(set(y_test))>1 else None

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
META_PATH.write_text(json.dumps({'accuracy':acc,'auc':auc}, indent=2))
print('Retrain complete', acc, auc)
