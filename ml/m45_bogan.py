from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

dateters = ['3/1/2021','3/2/2021','3/3/2021','3/4/2021','3/5/2021']
dates = pd.to_datetime(dateters)
print(dates)
print('=========================')

ts = Series([1, np.nan, np.nan, 8, 10], index=dates)
print(ts)

ts_intp_lineal = ts.interpolate() #결측지가 다 사라짐
print(ts_intp_lineal)