import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analytics import load_data, monthly_revenue_summary

bookings_path = project_root / 'bookings.xlsx'
revenue_path = project_root / 'revenue.xlsx'

print('Using project root:', project_root)
print('Bookings path:', bookings_path)
print('Revenue path:', revenue_path)

bookings, revenue = load_data(str(bookings_path), str(revenue_path))
print('\nRevenue columns:')
print(list(revenue.columns))

print('\nRevenue sample:')
print(revenue.head().to_string())

try:
    revenue_monthly = monthly_revenue_summary(revenue)
    print('\nmonthly_revenue_summary output sample:')
    print(revenue_monthly.head().to_string())
except Exception as e:
    print('\nmonthly_revenue_summary raised:', type(e).__name__, e)
    print('\nSuggestions:')
    print(' - Ensure revenue has a date column (like `date`, `created_at`, or `timestamp`).')
    print(' - Ensure revenue has a price column (like `price`, `amount`, `revenue`, or `calculated_price`).')
    print(' - If column names have leading/trailing spaces, try: revenue.columns = revenue.columns.str.strip()')
