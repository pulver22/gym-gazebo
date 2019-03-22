import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


# This function takes an array of numbers and smoothes them out.
# Smoothing is useful for making plots a little easier to read.
def sliding_mean(data_array, window=5):
    # data_array = array(data_array)
    new_list = []
    for i in range(len(data_array)):
        indices = range(max(i - window + 1, 0),
                        min(i + window + 1, len(data_array)))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return np.array(new_list)


folder ='/home/pulver/Desktop/avoidance/tb_log/run-'
firstTime=True
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
df_merged['std_dev'] = df_merged.std(axis=1)
# df_merged.to_csv(path_or_buf='/home/pulver/Desktop/final_monitor.csv', index=True)
#
#
a = np.arange(len(df_merged['mean']))

# mean = sliding_mean(df_merged['mean'], window=1000)
# std = sliding_mean(df_merged['std_dev'], window=1000)
df_merged['rolling_mean'] = df_merged['mean'].rolling(3500).mean()
df_merged['rolling_std'] = df_merged['std_dev'].rolling(3500).mean()
print(df_merged)
mean = df_merged['rolling_mean']
std = df_merged['rolling_std']

# TODO: the rolling method will create NaNs at the beginning of the plot. That should be fixed


# plot the std deviation as a transparent range at each training set size
# plt.fill_between(a, df_merged['mean'] - df_merged['std_dev'], df_merged['mean'] + df_merged['std_dev'], alpha=0.1, color="b")
plt.fill_between(a, mean - std, mean + std, alpha=0.1, color="b")

# sizes the window for readability and displays the plot
plt.xlim(min(a), max(a))
plt.ylim(-200, 200)



# plt.show(plt.plot(a, df_merged['mean']))
plt.show(plt.plot(a, mean))




