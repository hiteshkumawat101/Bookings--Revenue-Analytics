import sys
from pathlib import Path
proj = Path('c:/Users/dell7/OneDrive/Desktop/Assignment')
if str(proj) not in sys.path:
    sys.path.insert(0, str(proj))
import inspect
import src.analytics as a
print('module file:', a.__file__)
print('\n--- monthly_bookings_and_revenue source ---')
print(inspect.getsource(a.monthly_bookings_and_revenue))
