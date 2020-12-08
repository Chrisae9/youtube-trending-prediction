import pandas as pd
import numpy as np

df = pd.read_csv("data/USvideos.csv")
publish = df['publish_time']
trending = df['trending_date']
views = df['views']
row_count, col_count = df.shape
a_million = 1000000
label_arr = np.array([], dtype = np.int)

for i in range(row_count):
    pubNum = int(publish[i][8:10])
    trendNum = int(trending[i][3:5])
    viewsNum = int(views[i])
    if ((pubNum == trendNum or pubNum + 1 == trendNum) and viewsNum >= a_million):
        result = np.array([1], dtype = np.int)
    else:
        result = np.array([0], dtype = np.int)
    label_arr = np.append(label_arr, result)

np.savetxt('output.txt', label_arr, '%d', delimiter = ',')   
modified_df = df
modified_df['label'] = label_arr
