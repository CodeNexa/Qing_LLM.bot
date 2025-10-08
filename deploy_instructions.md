# Deploying to Hugging Face Spaces

1. Create a GitHub repo and push this project.
2. Create a Space on Hugging Face: choose Streamlit SDK and connect to your GitHub repo.
3. Add a secret HF_TOKEN in your GitHub repo (Settings -> Secrets) with a HF access token.
4. The GitHub Actions workflow will retrain and push updated model and trigger the Space rebuild.
