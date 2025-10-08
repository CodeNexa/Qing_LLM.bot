# backend/expert_rules.py
def apply_rules(row: dict):
    reasons = []
    raw = row.get('model_signal','HOLD')
    adjusted = raw
    if abs(row.get('return_5d',0)) > 0.15:
        adjusted = 'HOLD'
        reasons.append('High short-term volatility (>15% 5d)')
    if row.get('vol_10',0) < 10000:
        adjusted = 'HOLD'
        reasons.append('Low liquidity (vol_10 < 10000)')
    low = row.get('low',0); high = row.get('high',0)
    if low and (high-low)/max(low,1e-9) > 0.2:
        adjusted = 'HOLD'; reasons.append('Large intraday range (>20%)')
    if raw == 'BUY' and row.get('chg_%',0) > 2.5:
        adjusted = 'BUY_STRONG'; reasons.append('Momentum >2.5%')
    if row.get('llm_sentiment',0) < -0.7 and raw == 'BUY':
        adjusted = 'HOLD'; reasons.append('Strong negative sentiment override')
    return adjusted, reasons
