# Bookings & Revenue Analytics Project

This repository contains a complete data analytics workflow for bookings and revenue data, including:

- **Comprehensive Exploratory Data Analysis (EDA)**
- **Historical and Predictive Analytics**
- **Rule-based and Machine Learning Recommendations**
- **Interactive Streamlit Dashboard**
- **Jupyter Notebooks for Prototyping and Interview Preparation**

## File Structure

- `src/analytics.py` — Core analytics, forecasting, and recommendation logic
- `src/reporting.py` — Reporting and visualization helpers
- `src/app.py` — FastAPI backend for analytics API (optional)
- `streamlit_app.py` — Streamlit dashboard for interactive analysis
- `notebooks/eda_analysis.ipynb` — EDA template for interview/assignment prep
- `notebooks/exploratory.ipynb` — Exploratory analysis and prototyping
- `requirements.txt` — Python dependencies
- `bookings.xlsx`, `revenue.xlsx` — Example data files
- `scripts/diagnostic_bookings.py`, `scripts/diagnostic_revenue.py` — Diagnostic scripts

## Installation & How to Run

1. **Clone the repository** (or download the files):
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv env
   # On Windows:
   env\Scripts\activate
   # On Mac/Linux:
   source env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit dashboard:**
   ```bash
   streamlit run streamlit_app.py
   ```
   - The dashboard will open in your browser. Use the sidebar to load your data and explore analytics.

5. **Explore the Jupyter notebooks** for EDA and prototyping (open with Jupyter Lab/Notebook or VS Code).


## Notes
- Update file paths in notebooks and scripts as needed.
- Remove or anonymize any sensitive data before sharing publicly.
