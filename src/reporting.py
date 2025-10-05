import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from typing import Dict, Any

from .analytics import bookings_forecast, revenue_forecast


def generate_report(bookings_df: pd.DataFrame, revenue_df: pd.DataFrame, availability_hours_per_day: int = 9) -> Dict[str, Any]:
    """Generate analytics and forecasts from the provided DataFrames.

    Returns a dictionary with DataFrames and forecasts.
    """
    # Copy inputs
    b = bookings_df.copy()
    r = revenue_df.copy()

    # Ensure datetime parsing for common columns
    if 'booking_start_time' in b.columns:
        b['booking_start_time'] = pd.to_datetime(b['booking_start_time'])
    if 'booking_end_time' in b.columns:
        b['booking_end_time'] = pd.to_datetime(b['booking_end_time'])
    if 'created_at' in r.columns:
        r['created_at'] = pd.to_datetime(r['created_at'])
    if 'event_at' in r.columns:
        r['event_at'] = pd.to_datetime(r['event_at'])

    # Monthly aggregations
    b['month'] = b['booking_start_time'].dt.to_period('M')
    r['month'] = r['created_at'].dt.to_period('M')

    monthly_bookings = b.groupby('month').agg(
        total_bookings=('product_name', 'count'),
        avg_revenue=('calculated_price', 'mean'),
        total_revenue=('calculated_price', 'sum')
    ).reset_index()

    monthly_revenue = r.groupby('month').agg(
        total_revenue=('price', 'sum'),
        avg_price=('price', 'mean')
    ).reset_index()

    # Utilization
    b['booked_minutes'] = b.get('duration_in_mins', pd.Series(0)).fillna(0)
    room_util = b.groupby(['month', 'product_name']).agg(total_booked_minutes=('booked_minutes', 'sum')).reset_index()
    room_util['days_in_month'] = room_util['month'].dt.days_in_month
    minutes_per_day = availability_hours_per_day * 60
    room_util['available_minutes'] = room_util['days_in_month'] * minutes_per_day
    room_util['utilization_pct'] = room_util['total_booked_minutes'] / room_util['available_minutes'] * 100

    # Top tenants
    top_tenants = b.groupby('enterprise_name').agg(
        bookings=('product_name', 'count'),
        revenue=('calculated_price', 'sum')
    ).sort_values(by='revenue', ascending=False).reset_index().head(10)

    # Top products
    top_products = r.groupby('product_name').agg(
        total_revenue=('price', 'sum'),
        avg_price=('price', 'mean'),
        transactions=('product_name', 'count')
    ).sort_values(by='total_revenue', ascending=False).reset_index().head(10)

    # Forecasts using existing analytics helpers where possible
    try:
        fc_bookings = bookings_forecast(b, months=3)
    except Exception:
        fc_bookings = None
    try:
        fc_revenue = revenue_forecast(r, months=12)
    except Exception:
        fc_revenue = None

    # ROI products
    b['revenue_per_minute'] = b['calculated_price'] / b['booked_minutes'].replace(0, np.nan)
    roi_products = b.groupby('product_name').agg(
        avg_revenue_per_minute=('revenue_per_minute', 'mean'),
        total_revenue=('calculated_price', 'sum')
    ).sort_values(by='avg_revenue_per_minute', ascending=False).reset_index().head(10)

    # Underutilized rooms
    underutilized = room_util[room_util['utilization_pct'] < 30].copy()

    # Dynamic pricing suggestions
    def dynamic_pricing(row):
        if row['utilization_pct'] < 30:
            return 'Consider discounting this room to attract tenants'
        elif row['utilization_pct'] > 80:
            return 'Consider increasing price (high demand)'
        else:
            return 'Keep price stable'

    room_util['pricing_recommendation'] = room_util.apply(dynamic_pricing, axis=1)

    return {
        'monthly_bookings': monthly_bookings,
        'monthly_revenue': monthly_revenue,
        'room_utilization': room_util,
        'top_tenants': top_tenants,
        'top_products': top_products,
        'forecast_bookings': fc_bookings,
        'forecast_revenue': fc_revenue,
        'underutilized': underutilized,
        'roi_products': roi_products,
    }


def plot_monthly(monthly_bookings, monthly_revenue):
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    sns.barplot(data=monthly_bookings, x='month', y='total_bookings', ax=ax[0])
    ax[0].tick_params(axis='x', rotation=45)
    ax[0].set_title('Monthly Bookings')
    sns.lineplot(data=monthly_revenue, x='month', y='total_revenue', ax=ax[1])
    ax[1].tick_params(axis='x', rotation=45)
    ax[1].set_title('Monthly Revenue')
    plt.tight_layout()
    return fig
