import pandas as pd
from matplotlib import pyplot

series = pd.read_csv('trajectories/traj1.txt', index_col=0)
print(series.head())
series.plot()
pyplot.show()

upsampled = series.resample('D')