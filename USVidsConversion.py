import pandas as pd
import numpy as np
import re

def csv_conversion(file_name):
    df = pd.read_csv(file_name, encoding = "ISO-8859-1")
    publish = df['publish_time']
    trending = df['trending_date']
    views = df['views']
    titles = df['title']
    row_count = df.shape[0]
    a_million = 1000000
    label_arr = np.array([], dtype = np.int)
    clickbait_arr = np.array([], dtype = np.double)

    for i in range(row_count):
        pubNum = int(publish[i][8:10])
        trendNum = int(trending[i][3:5])
        viewsNum = int(views[i])
        if ((pubNum == trendNum or pubNum + 1 == trendNum) and viewsNum >= a_million):
            label_result = np.array([3], dtype = np.int)
        elif ((pubNum == trendNum or pubNum + 1 == trendNum) and viewsNum < a_million):
            label_result = np.array([2], dtype = np.int)
        elif (not(pubNum == trendNum or pubNum + 1 == trendNum) and viewsNum >= a_million):
            label_result = np.array([1], dtype = np.int)
        else:
            label_result = np.array([0], dtype = np.int)
        clickbait_result = np.array(clickbait(titles[i]), dtype = np.double)    
        clickbait_arr = np.append(clickbait_arr, clickbait_result)
        label_arr = np.append(label_arr, label_result)

    #np.savetxt('output.txt', label_arr, '%d', delimiter = ',')   
    modified_df = df
    modified_df['clickbait_score'] = clickbait_arr
    modified_df['label'] = label_arr
    return modified_df

#clickbait # of capital letters in title / total amount of letters
def clickbait(sentence):
    return len(re.findall(r'[A-Z]', sentence)) / len(sentence)

def averageClickbaitScore(dataframe):
    labels = dataframe['label']
    clickbait_scores = dataframe['clickbait_score']
    row_count = dataframe.shape[0]
    class0_arr = np.array([], dtype = np.double)
    class1_arr = np.array([], dtype = np.double)
    class2_arr = np.array([], dtype = np.double)
    class3_arr = np.array([], dtype = np.double)

    for i in range(row_count):
        result = np.array(clickbait_scores[i])
        if labels[i] == 0:
            class0_arr = np.append(class0_arr, result)
        elif labels[i] == 1:
            class1_arr = np.append(class1_arr, result)
        elif labels[i] == 2:
            class2_arr = np.append(class2_arr, result)
        else:
            class3_arr = np.append(class3_arr, result)
    
    result = np.array([], dtype = np.double)
    avg_class0 = np.array([np.average(class0_arr)], dtype = np.double)
    result = np.append(result, avg_class0)
    avg_class1 = np.array([np.average(class1_arr)], dtype = np.double)
    result = np.append(result, avg_class1)
    avg_class2 = np.array([np.average(class2_arr)], dtype = np.double)
    result = np.append(result, avg_class2)
    avg_class3 = np.array([np.average(class3_arr)], dtype = np.double)
    result = np.append(result, avg_class3)
    return result
    
df_US = csv_conversion("data/USvideos.csv")
#print(df_US)
#print(averageClickbaitScore(df_US))
#df_RU = csv_conversion("data/RUvideos.csv")
#df_MX = csv_conversion("data/MXvideos.csv")
#df_KR = csv_conversion("data/KRvideos.csv")
#df_JP = csv_conversion("data/JPvideos.csv")
#df_IN = csv_conversion("data/INvideos.csv")
#df_GB = csv_conversion("data/GBvideos.csv")
#df_FR = csv_conversion("data/FRvideos.csv")
#df_DE = csv_conversion("data/DEvideos.csv")
df_CA = csv_conversion("data/CAvideos.csv")

