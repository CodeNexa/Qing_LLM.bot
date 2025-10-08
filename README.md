# NSE LLM Bot — Full Hugging Face Ready

This repository contains a Streamlit-based NSE LLM Bot (paper-trading).
It includes:
- Local sentiment wrapper (backend/llm_wrapper.py)
- Expert rules (backend/expert_rules.py)
- Retrain script (backend/retrain_model.py)
- GitHub Actions workflow to retrain and trigger Hugging Face Space rebuild
- Streamlit app (app.py) for interactive use

Model metrics (on current dataset): accuracy=0.9878, auc=0.7388288516878894
