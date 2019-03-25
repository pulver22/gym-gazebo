import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


folder ='/home/pulver/Dropbox/University/PostDoc/Experiments/Avoidance/depth/' + 'run-'
rolling_window = 3500

#  https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html
datas=[]
df_final = pd.DataFrame()
for i in range(1,5):
    dataURI=folder+str(i)+'/monitor.csv'
    data= pd.read_csv(dataURI)
    data['experiment'] = str(i)
    data.set_index('Step', inplace=True)
    df2 = data.reindex(np.arange(0, 103000))
    df2 = df2.ffill()
    df2 = df2.bfill()
    del df2['Wall time']
    del df2['experiment']
    df2.to_csv(path_or_buf=folder+str(i)+'/monitor_2.csv')
    datas.append(df2)


df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Step'], how='outer'), datas)
df_merged['mean'] = df_merged.mean(axis=1)
df_merged['median'] = df_merged.median(axis=1)
df_merged['std_dev'] = df_merged.std(axis=1)
# df_merged.to_csv(path_or_buf='/home/pulver/Desktop/final_monitor.csv', index=True)
#
#
a = np.arange(len(df_merged['mean']))

# mean = sliding_mean(df_merged['mean'], window=1000)
# std = sliding_mean(df_merged['std_dev'], window=1000)
df_merged['rolling_mean'] = df_merged['mean'].rolling(rolling_window).mean()
df_merged['rolling_median'] = df_merged['median'].rolling(rolling_window).mean()
df_merged['rolling_std'] = df_merged['std_dev'].rolling(rolling_window).mean()

# The rolling window will create NaN at the beginning of the plot, so we must fill them in some way
# TODO: replace bfill() with the original values before applying the rolling method
df_merged = df_merged.bfill()

# Calculate statistics
mean = df_merged['rolling_mean']
median = df_merged['rolling_median']
std = df_merged['rolling_std']

# Plot the std deviation as a transparent range at each training set size
plt.fill_between(a, mean - std, mean + std, alpha=0.1, color="b")

# Sizes the window for readability and displays the plot
plt.xlim(min(a), max(a))
plt.ylim(-200, 200)

plt.plot(a, mean)
plt.plot(a, median)

plt.show()




