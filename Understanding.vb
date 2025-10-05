Project Architecture Overview
1. Data Sources
bookings.xlsx: Contains historical booking data (room, tenant, time, price, etc.).
revenue.xlsx: Contains historical revenue data (product, price, date, etc.).
2. Data Ingestion & Preprocessing
Data is loaded using the load_data function in analytics.py.
Column names are normalized for consistency.
Dates are parsed and missing values are handled.
3. Analytics & Feature Engineering
Monthly Aggregations: Functions like monthly_bookings_and_revenue and monthly_revenue_summary compute monthly trends for bookings and revenue.
Utilization Metrics: utilization_metrics calculates the percentage of time each room is booked, identifying underutilized rooms.
Top Tenants & Products: Functions identify the most valuable tenants and products.
4. Predictive Modeling
Forecasting: Uses Facebook Prophet (via bookings_forecast and revenue_forecast) to predict future bookings and revenue.
Outputs include expected value (yhat), best case (yhat_upper), and worst case (yhat_lower).
Visualization: Forecasts are visualized with Plotly or Seaborn, showing all three scenarios.
5. Recommendation System
Rule-Based Recommendations:
Underutilized rooms: Identified by a utilization threshold (e.g., <30%).
High ROI products: Products with the highest average revenue per minute.
Machine Learning-Based Recommendations:
Clustering (KMeans): Groups rooms by utilization to find clusters of underutilized rooms.
Association Rule Mining (Apriori): Finds product combinations frequently booked together, suggesting upsell opportunities.
6. Application Layer
Streamlit Dashboard:
Sidebar for data upload.
Tabs/sections for historical analysis, forecasts, and recommendations.
Interactive controls for forecast period and utilization threshold.
Professional markdown explanations for clarity.
7. Code Structure
analytics.py: Core analytics, forecasting, and recommendation logic.
reporting.py: (Optional) Generates summary reports and plots.
streamlit_app.py: User interface and workflow orchestration.
notebooks: Jupyter notebooks for exploratory analysis and prototyping.
scripts: Diagnostic scripts for debugging and validation.
8. Libraries Used
pandas, numpy: Data manipulation.
prophet: Time series forecasting.
scikit-learn: Clustering (KMeans).
mlxtend: Association rule mining (Apriori).
matplotlib, seaborn, plotly: Visualization.
streamlit: Web dashboard.
Workflow Summary
User loads data via the Streamlit sidebar.
Historical metrics are computed and visualized.
Forecasts for bookings and revenue are generated, with best/worst case scenarios.
Recommendations are provided using both business rules and machine learning.
All results are presented in a clear, interactive dashboard with professional explanations.
Business Value
Enables data-driven decision making for room utilization, product focus, and revenue planning.
Combines explainable business rules with advanced ML for robust recommendations.
Professional presentation suitable for stakeholders and non-technical users.

--------------------------------------------------------------------------------------------------------------------------------

1. analytics.py
Purpose: Core analytics, forecasting, and recommendation logic.
Key Functions:
load_data: Loads and cleans bookings and revenue Excel files.
monthly_bookings_and_revenue, monthly_revenue_summary: Aggregate bookings/revenue by month, room, or product.
utilization_metrics: Calculates room utilization as a percentage.
top_tenants, top_revenue_products: Finds top tenants and products by revenue/bookings.
bookings_forecast, revenue_forecast: Uses Prophet to forecast future bookings/revenue, returning expected, upper, and lower bounds.
recommend_underutilized, high_roi_products: Rule-based recommendations for underutilized rooms and high ROI products.
ml_underutilized_rooms: Uses KMeans clustering to find groups of underutilized rooms.
ml_product_upsell: Uses Apriori association rules to suggest product upsell opportunities.
2. reporting.py
Purpose: Generates summary reports and visualizations (optional, for advanced reporting).
Key Functions:
generate_report: Aggregates all analytics, forecasts, and recommendations into a single dictionary for reporting.
plot_monthly: Plots monthly bookings and revenue using Seaborn/Matplotlib.
3. streamlit_app.py
Purpose: User interface/dashboard for the analytics and recommendations.
Key Logic:
Loads data from user input (sidebar).
Displays historical summaries (bookings, revenue, utilization, top tenants/products).
Shows both rule-based and ML-based recommendations.
Runs and visualizes forecasts for bookings and revenue, including best/worst case.
Provides markdown explanations and a professional summary for clarity.
4. exploratory.ipynb & exploratory_partial.ipynb
Purpose: Jupyter notebooks for data exploration and prototyping.
Key Logic:
Load and inspect data.
Run analytics and forecasting functions.
Visualize results and debug issues interactively.
5. diagnostic_bookings.py & diagnostic_revenue.py
Purpose: Diagnostic scripts for debugging and validating data and analytics functions.
Key Logic:
Load data and print sample rows/columns.
Run aggregation functions and print results or helpful error messages.
6. app.py
Purpose: FastAPI backend for serving analytics as an API (optional, for integration or automation).
Key Logic:
Defines API endpoints for summary analytics, forecasts, and recommendations.
Uses the same analytics functions as the Streamlit app.
7. requirements.txt
Purpose: Lists all required Python packages for the project (pandas, numpy, prophet, scikit-learn, mlxtend, streamlit, etc.).
8. __init__.py
Purpose: Marks the src directory as a Python package.
If you want a deeper dive into any specific function or file, let me know!













---------------------MAIN logic
Historical Analysis
Monthly Aggregation: Functions group data by month (and optionally by room or product) to calculate:
Total bookings and revenue per month.
Average revenue per booking or product.
Utilization Metrics: Calculates the percentage of time each room is booked, helping to identify underutilized rooms.
Top Tenants/Products: Aggregates bookings and revenue to find the most valuable tenants and products.
3. Predictive Analysis (Forecasting)
Uses the Prophet library to forecast future bookings and revenue.
The forecast includes:
Expected value (yhat)
Best case (yhat_upper)
Worst case (yhat_lower)
Results are visualized as tables and plots for easy interpretation.
4. Recommendations
Rule-Based:
Underutilized rooms are identified by a utilization threshold (e.g., <30%).
High ROI products are those with the highest average revenue per minute.
Machine Learning-Based:
Clustering (KMeans): Groups rooms by utilization to find clusters of underutilized rooms.
Association Rules (Apriori): Finds product combinations that are frequently booked together, suggesting upsell opportunities.
5. Presentation
All results are displayed in a Streamlit dashboard with clear explanations and interactive controls.
Notebooks are used for prototyping and sharing analysis steps.
Summary:
The code automates the process of loading, cleaning, analyzing, forecasting, and recommending actions based on bookings and revenue data. It combines business logic with machine learning to provide both explainable and data-driven insights for decision-making.