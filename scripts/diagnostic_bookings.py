import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analytics import load_data, monthly_bookings_and_revenue

bookings_path = project_root / 'bookings.xlsx'
revenue_path = project_root / 'revenue.xlsx'

print('Using project root:', project_root)
print('Bookings path:', bookings_path)

bookings, revenue = load_data(str(bookings_path), str(revenue_path))
print('\nBookings columns:')
print(list(bookings.columns))

print('\nBookings sample:')
print(bookings.head().to_string())

try:
    bookings_monthly = monthly_bookings_and_revenue(bookings)
    print('\nmonthly_bookings_and_revenue output sample:')
    print(bookings_monthly.head().to_string())
except Exception as e:
    print('\nmonthly_bookings_and_revenue raised:', type(e).__name__, e)
    print('\nSuggestions:')
    print(' - Ensure bookings has a start datetime column (like `start_time`, `start`, `start_datetime`, or `check_in`).')
    print(' - If the column is named differently, either rename it or update the analytics helper to include that name.')
    print(' - If columns have extra spaces, run: bookings.columns = bookings.columns.str.strip()')
