import streamlit as st
import pandas as pd
from pathlib import Path
try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

import matplotlib.pyplot as plt
import seaborn as sns

from src.analytics import (
    load_data,
    monthly_bookings_and_revenue,
    monthly_revenue_summary,
    utilization_metrics,
    top_tenants,
    top_revenue_products,
    bookings_forecast,
    revenue_forecast,
    recommend_underutilized,
    high_roi_products,
    ml_underutilized_rooms,
    ml_product_upsell,
)

st.set_page_config(page_title='Bookings Analytics', layout='wide')


# Professional summary for data science submission
st.title('Bookings & Revenue Analytics')
st.markdown("""
## Project Summary
This dashboard provides historical and predictive analytics for bookings and revenue, with actionable recommendations for business growth. It combines rule-based and machine learning approaches to deliver:
- **Historical analysis**: Trends in bookings and revenue, room utilization, top tenants, and high-performing products.
- **Predictive analysis**: Forecasts for bookings (next quarter) and revenue (next year), including best and worst case scenarios.
- **Recommendations**: Identification of underutilized rooms and upsell opportunities, using both business rules and machine learning (KMeans clustering, association rules).

**How to use:**
1. Load your data from the sidebar.
2. Explore historical trends and recommendations.
3. Run forecasts and review actionable insights for strategic planning.
""")

st.sidebar.header('Data')
bookings_path = st.sidebar.text_input('Bookings Excel path', value=str(Path.cwd() / 'bookings.xlsx'))
revenue_path = st.sidebar.text_input('Revenue Excel path', value=str(Path.cwd() / 'revenue.xlsx'))

if st.sidebar.button('Load data'):
    try:
        bookings, revenue = load_data(bookings_path, revenue_path)
        st.success(f'Loaded {len(bookings)} bookings and {len(revenue)} revenue rows')
        st.session_state['bookings'] = bookings
        st.session_state['revenue'] = revenue
    except Exception as e:
        st.error(str(e))

if 'bookings' in st.session_state and 'revenue' in st.session_state:
    bookings = st.session_state['bookings']
    revenue = st.session_state['revenue']

    st.header('Historical Summaries')
    st.markdown("""
    ### Historical Analysis
    - **Monthly Bookings**: Track booking volume and seasonality.
    - **Monthly Revenue**: Monitor revenue trends over time.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Monthly Bookings')
        bm = monthly_bookings_and_revenue(bookings)
        st.dataframe(bm)
        if _HAS_PLOTLY:
            fig = px.bar(bm, x='month', y='bookings_count', color=bm.columns[1] if 'room_id' in bm.columns else None)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=bm, x='month', y='bookings_count', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
    with col2:
        st.subheader('Monthly Revenue')
        rm = monthly_revenue_summary(revenue)
        st.dataframe(rm)
        if _HAS_PLOTLY:
            fig2 = px.line(rm, x='month', y='total_revenue')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=rm, x='month', y='total_revenue', ax=ax2)
            plt.xticks(rotation=45)
            st.pyplot(fig2)

    st.header('Utilization & Recommendations')
    st.markdown("""
    ### Recommendations
    - **Underutilized Rooms**: Identify rooms with low usage for targeted promotions.
    - **High ROI Products**: Focus on products generating the most revenue per minute.
    - **ML-based Clustering**: Discover hidden patterns in room utilization.
    - **ML-based Upsell**: Suggest product combinations likely to increase sales.
    """)
    util = utilization_metrics(bookings)
    st.subheader('Utilization (sample)')
    st.dataframe(util.head())

    st.subheader('Underutilized Rooms')
    under = recommend_underutilized(util, threshold_pct=st.slider('Threshold %', 0, 100, 30))
    st.dataframe(under)

    st.subheader('High ROI Products')
    roi = high_roi_products(revenue)
    st.dataframe(roi.head(20))

    st.subheader('ML-based Underutilized Room Clusters')
    ml_under = ml_underutilized_rooms(util)
    st.dataframe(ml_under)

    st.subheader('ML-based Product Upsell Suggestions')
    ml_upsell = ml_product_upsell(bookings)
    st.dataframe(ml_upsell.head(20))

    st.header('Forecasts')
    months_book = st.number_input('Months for bookings forecast', min_value=1, max_value=12, value=3)
    if st.button('Run bookings forecast'):
        fc_book = bookings_forecast(bookings, months=months_book)
        df_fc_book = pd.DataFrame(fc_book)
        st.markdown("""
        **Bookings Forecast Table (Next Quarter)**
        - `yhat`: Predicted bookings (expected case)
        - `yhat_upper`: Best case (upper bound)
        - `yhat_lower`: Worst case (lower bound)
        """)
        st.dataframe(df_fc_book.tail(20))
        if _HAS_PLOTLY:
            fig_book = px.line(df_fc_book, x='ds', y='yhat', title='Bookings Forecast')
            fig_book.add_scatter(x=df_fc_book['ds'], y=df_fc_book['yhat_upper'], mode='lines', name='Best Case (Upper)')
            fig_book.add_scatter(x=df_fc_book['ds'], y=df_fc_book['yhat_lower'], mode='lines', name='Worst Case (Lower)')
            st.plotly_chart(fig_book)
        else:
            fig_book, ax_book = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=df_fc_book, x='ds', y='yhat', label='Expected', ax=ax_book)
            sns.lineplot(data=df_fc_book, x='ds', y='yhat_upper', label='Best Case', ax=ax_book)
            sns.lineplot(data=df_fc_book, x='ds', y='yhat_lower', label='Worst Case', ax=ax_book)
            plt.xticks(rotation=45)
            ax_book.set_title('Bookings Forecast: Expected, Best, and Worst Case')
            plt.legend()
            st.pyplot(fig_book)

    months_rev = st.number_input('Months for revenue forecast', min_value=1, max_value=24, value=12)
    if st.button('Run revenue forecast'):
        fc_rev = revenue_forecast(revenue, months=months_rev)
        df_fc = pd.DataFrame(fc_rev)
        st.markdown("""
        **Revenue Forecast Table (Next Year)**
        - `yhat`: Predicted revenue (expected case)
        - `yhat_upper`: Best case (upper bound)
        - `yhat_lower`: Worst case (lower bound)
        """)
        st.dataframe(df_fc.tail(20))
        if _HAS_PLOTLY:
            fig3 = px.line(df_fc, x='ds', y='yhat', title='Revenue Forecast')
            fig3.add_scatter(x=df_fc['ds'], y=df_fc['yhat_upper'], mode='lines', name='Best Case (Upper)')
            fig3.add_scatter(x=df_fc['ds'], y=df_fc['yhat_lower'], mode='lines', name='Worst Case (Lower)')
            st.plotly_chart(fig3)
        else:
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=df_fc, x='ds', y='yhat', label='Expected', ax=ax3)
            sns.lineplot(data=df_fc, x='ds', y='yhat_upper', label='Best Case', ax=ax3)
            sns.lineplot(data=df_fc, x='ds', y='yhat_lower', label='Worst Case', ax=ax3)
            plt.xticks(rotation=45)
            ax3.set_title('Revenue Forecast: Expected, Best, and Worst Case')
            plt.legend()
            st.pyplot(fig3)

else:
    st.info('Please load data from the sidebar to begin.')
