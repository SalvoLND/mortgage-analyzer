# Mortgage Analyzer

A Streamlit web app for calculating, comparing and visualising mortgage
scenarios — with an optional DeepSeek-powered AI assistant that can answer
questions about the numbers currently on screen.

## Features

- **📊 Basic Analysis** — monthly payment, total interest, full amortization
  schedule with interactive charts and CSV download
- **💰 Advanced Features** — extra-payment impact and payoff comparisons
- **🔄 Refinance Calculator** — side-by-side comparison of current vs. new terms
- **⚡ Early Repayment Simulator** — model lump sums and overpayments
- **🤖 Ask AI** — chat about your own scenario, powered by the DeepSeek API
  (optional; the four calculator tabs work fully without a key)

## Running locally

```bash
pip install -r requirements.txt
streamlit run app_ai.py
```

The app opens at `http://localhost:8501`.

> `app.py` is the original calculator-only version and is kept for reference.
> `app_ai.py` is the entry point and is a superset of it.

## Configuring the AI assistant (optional)

The app never contains an API key. At runtime it looks for a DeepSeek key in
three places, in order:

1. The key box in the sidebar
2. `st.secrets["DEEPSEEK_API_KEY"]`
3. The `DEEPSEEK_API_KEY` environment variable

For local use, either export it:

```bash
export DEEPSEEK_API_KEY="sk-your-key-here"
```

or create `.streamlit/secrets.toml` next to `app_ai.py`:

```toml
DEEPSEEK_API_KEY = "sk-your-key-here"
```

`.streamlit/secrets.toml` is gitignored — keep it that way.

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), choose **New app** and
   select this repo and branch.
3. Set **Main file path** to `app_ai.py`.
4. Under **Advanced settings → Secrets**, paste:

   ```toml
   DEEPSEEK_API_KEY = "sk-your-key-here"
   ```

   Skip this if you'd rather have users supply their own key in the sidebar.
5. Deploy. Dependencies are installed from `requirements.txt`.

## Requirements

- Python 3.9+
- streamlit ≥ 1.46.1, pandas, plotly, streamlit-aggrid, requests

See [AI_ASSISTANT_DOCS.md](AI_ASSISTANT_DOCS.md) for the AI assistant's
architecture, prompt design and error handling.

## Tests

```bash
python -m unittest test_ai_assistant.py
```
