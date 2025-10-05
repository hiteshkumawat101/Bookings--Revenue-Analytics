from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from analytics import (
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
)

app = FastAPI(title='Bookings Analytics API')


class Paths(BaseModel):
    bookings_path: str
    revenue_path: str


@app.post('/analytics/summary')
def analytics_summary(paths: Paths):
    try:
        bookings, revenue = load_data(paths.bookings_path, paths.revenue_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    bookings_monthly = monthly_bookings_and_revenue(bookings)
    revenue_monthly = monthly_revenue_summary(revenue)
    util = utilization_metrics(bookings)
    tenants = top_tenants(bookings)
    products = top_revenue_products(revenue)

    return {
        'bookings_monthly': bookings_monthly.to_dict(orient='records'),
        'revenue_monthly': revenue_monthly.to_dict(orient='records'),
        'utilization': util.to_dict(orient='records'),
        'top_tenants': tenants.to_dict(orient='records'),
        'top_products': products.to_dict(orient='records'),
    }


@app.post('/forecast/bookings')
def forecast_bookings(paths: Paths, months: Optional[int] = 3):
    bookings, _ = load_data(paths.bookings_path, paths.revenue_path)
    fc = bookings_forecast(bookings, months=months)
    return fc


@app.post('/forecast/revenue')
def forecast_revenue(paths: Paths, months: Optional[int] = 12):
    _, revenue = load_data(paths.bookings_path, paths.revenue_path)
    fc = revenue_forecast(revenue, months=months)
    return fc


@app.post('/recommendations/rooms')
def recommendations_rooms(paths: Paths, threshold_pct: Optional[float] = 30.0):
    bookings, _ = load_data(paths.bookings_path, paths.revenue_path)
    util = utilization_metrics(bookings)
    under = recommend_underutilized(util, threshold_pct=threshold_pct)
    return under.to_dict(orient='records')


@app.post('/recommendations/products')
def recommendations_products(paths: Paths):
    _, revenue = load_data(paths.bookings_path, paths.revenue_path)
    agg = high_roi_products(revenue)
    return agg.to_dict(orient='records')
