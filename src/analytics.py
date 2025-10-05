import numpy as np
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.metrics.pairwise import cosine_similarity


def _try_parse_datetime(df, candidates):
    for c in candidates:
        if c in df.columns:
            return pd.to_datetime(df[c])
    return None


def load_data(bookings_path: str, revenue_path: str):
    """Load bookings and revenue Excel files into DataFrames.

    This function is resilient to several common column name variations.
    """
    bookings = pd.read_excel(bookings_path)
    revenue = pd.read_excel(revenue_path)

    # normalize column names to lowercase
    bookings.columns = [c.strip() for c in bookings.columns]
    revenue.columns = [c.strip() for c in revenue.columns]

    return bookings, revenue


def monthly_bookings_and_revenue(bookings: pd.DataFrame):
    """Group bookings by month + room/property -> count & avg revenue.

    Returns two DataFrames: bookings_by_month_room and revenue_by_month.
    """
    df = bookings.copy()

    # find datetime column
    start = _try_parse_datetime(df, ['start_time', 'start', 'start_datetime', 'check_in', 'booking_start_time', 'booking_start'])
    if start is None:
        raise ValueError('Could not find start datetime column in bookings')
    df['start_dt'] = start
    df['month'] = df['start_dt'].dt.to_period('M').dt.to_timestamp()

    price_col = None
    for c in ['calculated_price', 'price', 'amount', 'revenue']:
        if c in df.columns:
            price_col = c
            break

    if price_col is None:
        df['calculated_price'] = 0.0
        price_col = 'calculated_price'

    # group by month and room/property if columns exist
    room_cols = []
    if 'room_id' in df.columns:
        room_cols.append('room_id')
    if 'property_id' in df.columns:
        room_cols.append('property_id')

    if room_cols:
        grp = ['month'] + room_cols
        bookings_by_month_room = (
            df.groupby(grp)
            .agg(bookings_count=('start_dt', 'count'), avg_revenue=(price_col, 'mean'))
            .reset_index()
        )
    else:
        bookings_by_month_room = (
            df.groupby('month').agg(bookings_count=('start_dt', 'count'), avg_revenue=(price_col, 'mean')).reset_index()
        )

    return bookings_by_month_room


def monthly_revenue_summary(revenue: pd.DataFrame):
    df = revenue.copy()
    date = _try_parse_datetime(df, ['date', 'created_at', 'timestamp'])
    if date is None:
        # try to infer from bookings-like columns
        if 'month' in df.columns:
            df['month'] = pd.to_datetime(df['month'])
        else:
            raise ValueError('Could not find date column in revenue')
    else:
        df['month'] = pd.to_datetime(date).dt.to_period('M').dt.to_timestamp()

    price_col = None
    for c in ['price', 'amount', 'revenue', 'calculated_price']:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError('Could not find price column in revenue')

    summary = df.groupby('month').agg(total_revenue=(price_col, 'sum'), avg_revenue=(price_col, 'mean')).reset_index()
    return summary


def utilization_metrics(bookings: pd.DataFrame, available_start_hour=9, available_end_hour=18):
    """Calculate utilization % per room per month.

    Booked minutes are computed from start/end columns (common names handled).
    """
    df = bookings.copy()
    start = _try_parse_datetime(df, ['start_time', 'start', 'start_datetime', 'check_in', 'booking_start_time', 'booking_start'])
    end = _try_parse_datetime(df, ['end_time', 'end', 'end_datetime', 'check_out', 'booking_end_time', 'booking_end'])
    if start is None or end is None:
        raise ValueError('Could not find start/end datetime columns in bookings')
    df['start_dt'] = start
    df['end_dt'] = end
    df['month'] = df['start_dt'].dt.to_period('M').dt.to_timestamp()

    # compute booked minutes clipped to available window per day
    def clip_minutes(row):
        s = row['start_dt']
        e = row['end_dt']
        # clip to same-day available window for simplicity
        s_clip = max(s, s.replace(hour=available_start_hour, minute=0, second=0))
        e_clip = min(e, e.replace(hour=available_end_hour, minute=0, second=0))
        delta = (e_clip - s_clip).total_seconds() / 60.0
        return max(delta, 0.0)

    df['booked_mins'] = df.apply(clip_minutes, axis=1)

    if 'room_id' in df.columns:
        grp = ['month', 'room_id']
    elif 'property_id' in df.columns:
        grp = ['month', 'property_id']
    else:
        grp = ['month']

    booked = df.groupby(grp).agg(booked_minutes=('booked_mins', 'sum')).reset_index()

    # available minutes per month
    # compute number of days in month and multiply by available minutes per day
    def month_available_minutes(month_ts):
        month_start = pd.to_datetime(month_ts)
        next_month = (month_start + pd.offsets.MonthBegin(1))
        days = (next_month - month_start).days
        minutes_per_day = (available_end_hour - available_start_hour) * 60
        return days * minutes_per_day

    booked['available_minutes'] = booked['month'].apply(month_available_minutes)
    booked['utilization_pct'] = 100.0 * booked['booked_minutes'] / booked['available_minutes']
    return booked


def top_tenants(bookings: pd.DataFrame, top_n=10):
    df = bookings.copy()
    price_col = None
    for c in ['calculated_price', 'price', 'amount', 'revenue']:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        df['calculated_price'] = 0.0
        price_col = 'calculated_price'

    if 'enterprise_name' not in df.columns:
        raise ValueError('enterprise_name column required for top_tenants')

    agg = df.groupby('enterprise_name').agg(booking_count=('enterprise_name', 'count'), total_revenue=(price_col, 'sum')).reset_index()
    agg = agg.sort_values(['booking_count', 'total_revenue'], ascending=[False, False]).head(top_n)
    return agg


def top_revenue_products(revenue: pd.DataFrame, top_n=20):
    df = revenue.copy()
    price_col = None
    for c in ['price', 'amount', 'revenue', 'calculated_price']:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError('Price column not found in revenue')

    group_cols = []
    if 'product_name' in df.columns:
        group_cols.append('product_name')
    if 'product_type' in df.columns:
        group_cols.append('product_type')

    if not group_cols:
        raise ValueError('product_name or product_type required in revenue for top products')

    agg = df.groupby(group_cols).agg(total_revenue=(price_col, 'sum'), avg_revenue=(price_col, 'mean'), count=('product_name', 'count' if 'product_name' in df.columns else price_col)).reset_index()
    agg = agg.sort_values('total_revenue', ascending=False).head(top_n)
    return agg


def forecast_series(df, date_col='ds', value_col='y', periods=12, freq='M'):
    m = Prophet()
    m.fit(df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'}))
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


def bookings_forecast(bookings: pd.DataFrame, months=3, group_by='room_id'):
    """Forecast next `months` of bookings per room/property using Prophet.

    If group_by is not present, aggregate overall.
    """
    df = bookings.copy()
    start = _try_parse_datetime(df, ['start_time', 'start', 'start_datetime', 'check_in', 'booking_start_time', 'booking_start'])
    if start is None:
        raise ValueError('Could not find start datetime for forecasting')
    df['month'] = pd.to_datetime(start).dt.to_period('M').dt.to_timestamp()

    if group_by in df.columns:
        results = {}
        for key, g in df.groupby(group_by):
            ts = g.groupby('month').size().reset_index(name='count')
            ts = ts.rename(columns={'month': 'ds', 'count': 'y'})
            if len(ts) < 2:
                continue
            fc = forecast_series(ts, periods=months, freq='M')
            results[str(key)] = fc.to_dict(orient='list')
        return results
    else:
        ts = df.groupby('month').size().reset_index(name='count')
        ts = ts.rename(columns={'month': 'ds', 'count': 'y'})
        fc = forecast_series(ts, periods=months, freq='M')
        return fc.to_dict(orient='list')


def revenue_forecast(revenue: pd.DataFrame, months=12):
    df = revenue.copy()
    date = _try_parse_datetime(df, ['date', 'created_at', 'timestamp'])
    if date is None:
        raise ValueError('Could not find date column in revenue for forecasting')
    df['month'] = pd.to_datetime(date).dt.to_period('M').dt.to_timestamp()
    price_col = None
    for c in ['price', 'amount', 'revenue', 'calculated_price']:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError('Price column not found in revenue')

    ts = df.groupby('month').agg(total_revenue=(price_col, 'sum')).reset_index()
    ts = ts.rename(columns={'month': 'ds', 'total_revenue': 'y'})
    fc = forecast_series(ts, periods=months, freq='M')
    return fc.to_dict(orient='list')


def recommend_underutilized(util_df: pd.DataFrame, threshold_pct=30.0):
    """Return rooms with utilization < threshold_pct.
    util_df is expected to be the output of utilization_metrics.
    """
    under = util_df[util_df['utilization_pct'] < threshold_pct].copy()
    under = under.sort_values('utilization_pct')
    return under


def high_roi_products(revenue: pd.DataFrame):
    df = revenue.copy()
    # Try to find columns for product and price
    product_col = None
    for c in ['product_name', 'room_id', 'property_id']:
        if c in df.columns:
            product_col = c
            break
    price_col = None
    for c in ['price', 'amount', 'revenue', 'calculated_price']:
        if c in df.columns:
            price_col = c
            break
    duration_col = None
    for c in ['duration_in_mins', 'booked_minutes', 'duration', 'minutes']:
        if c in df.columns:
            duration_col = c
            break
    if product_col is None or price_col is None:
        return pd.DataFrame()
    # If duration is available, use it for ROI; else just use total revenue
    if duration_col:
        df['revenue_per_minute'] = df[price_col] / df[duration_col].replace(0, np.nan)
        agg = df.groupby(product_col).agg(
            avg_revenue_per_minute=('revenue_per_minute', 'mean'),
            total_revenue=(price_col, 'sum'),
            transactions=(product_col, 'count')
        ).sort_values(by='avg_revenue_per_minute', ascending=False).reset_index()
    else:
        agg = df.groupby(product_col).agg(
            total_revenue=(price_col, 'sum'),
            transactions=(product_col, 'count')
        ).sort_values(by='total_revenue', ascending=False).reset_index()
    return agg

# --- ML-based Recommendations ---
def ml_underutilized_rooms(util_df: pd.DataFrame, n_clusters=2):
    """Cluster rooms by utilization to find underutilized clusters."""
    # Assume util_df has columns: ['room_id', 'month', 'utilization_pct']
    if 'room_id' not in util_df.columns or 'utilization_pct' not in util_df.columns:
        return util_df
    # Aggregate by room
    room_util = util_df.groupby('room_id').agg(avg_utilization=('utilization_pct', 'mean')).reset_index()
    X = room_util[['avg_utilization']].fillna(0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    room_util['cluster'] = kmeans.fit_predict(X)
    # Find cluster with lowest mean utilization
    cluster_means = room_util.groupby('cluster')['avg_utilization'].mean()
    underutilized_cluster = cluster_means.idxmin()
    room_util['ml_underutilized'] = room_util['cluster'] == underutilized_cluster
    return room_util[room_util['ml_underutilized']]

def ml_product_upsell(bookings: pd.DataFrame, min_support=0.05, min_confidence=0.3):
    """Suggest product upsells using association rule mining (Apriori)."""
    # Assume bookings has columns: ['enterprise_name', 'product_name']
    if 'enterprise_name' not in bookings.columns or 'product_name' not in bookings.columns:
        return pd.DataFrame()
    # Build basket: list of products per enterprise
    basket = bookings.groupby('enterprise_name')['product_name'].apply(list).tolist()
    te = TransactionEncoder()
    te_ary = te.fit_transform(basket)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    freq = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    # Only keep rules with length 1 antecedents (simple upsell)
    rules = rules[rules['antecedents'].apply(lambda x: len(x) == 1)]
    # Format for display
    rules['antecedent'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['consequent'] = rules['consequents'].apply(lambda x: list(x)[0])
    return rules[['antecedent', 'consequent', 'support', 'confidence', 'lift']].sort_values('confidence', ascending=False)
    df = revenue.copy()
    price_col = None
    for c in ['price', 'amount', 'revenue', 'calculated_price']:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError('price column required')

    # calculate revenue per booking minute if minutes column exists
    if 'booking_minutes' in df.columns:
        df['rev_per_min'] = df[price_col] / df['booking_minutes']
    else:
        df['rev_per_min'] = df[price_col]

    group_cols = []
    if 'product_name' in df.columns:
        group_cols.append('product_name')
    if 'product_type' in df.columns:
        group_cols.append('product_type')
    if not group_cols:
        raise ValueError('product_name or product_type required')

    agg = df.groupby(group_cols).agg(total_revenue=(price_col, 'sum'), avg_rev_per_min=('rev_per_min', 'mean')).reset_index()
    agg = agg.sort_values('avg_rev_per_min', ascending=False)
    return agg
