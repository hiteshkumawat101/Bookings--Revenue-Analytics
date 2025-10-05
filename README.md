# Bookings Analytics

This workspace contains a small analytics package, a FastAPI app and an exploratory notebook for computing historical analytics, forecasts, and recommendations from `bookings.xlsx` and `revenue.xlsx`.

Files added:
- `src/analytics.py` - core analysis functions (loading, grouping, utilization, forecasts, recommendations)
- `src/app.py` - FastAPI service exposing endpoints described in the prompt
- `requirements.txt` - Python dependencies
- `notebooks/exploratory.ipynb` - notebook demonstrating EDA and forecasting (opens locally)

Quickstart (Windows PowerShell):

1. Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the API:

```powershell
uvicorn src.app:app --reload --port 8000
```

3. Open the notebook in Jupyter or VS Code and run cells to reproduce analytics.

Notes:
- The code attempts to be robust to common column name variations, but you may need to adapt column names to your exact files.
- Prophet package may be installed as `prophet` or `fbprophet` depending on environment; adjust `requirements.txt` if needed.
