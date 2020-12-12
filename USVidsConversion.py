import pandas as pd
import numpy as np

def csv_conversion(file_name):
    df = pd.read_csv(file_name, encoding = "ISO-8859-1")
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

    #np.savetxt('output.txt', label_arr, '%d', delimiter = ',')   
    modified_df = df
    modified_df['label'] = label_arr
    return modified_df

df_US = csv_conversion("data/USvideos.csv")
df_RU = csv_conversion("data/RUvideos.csv")
df_MX = csv_conversion("data/MXvideos.csv")
df_KR = csv_conversion("data/KRvideos.csv")
df_JP = csv_conversion("data/JPvideos.csv")
df_IN = csv_conversion("data/INvideos.csv")
df_GB = csv_conversion("data/GBvideos.csv")
df_FR = csv_conversion("data/FRvideos.csv")
df_DE = csv_conversion("data/DEvideos.csv")
df_CA = csv_conversion("data/CAvideos.csv")