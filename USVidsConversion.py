import pandas as pd
import numpy as np
import re
from PIL import Image
import requests
import pytesseract
import io
from dateutil import parser
from pandarallel import pandarallel

pandarallel.initialize()

def getLabel(publish, trending, views):
    
    # switch the year and the day
    trending = trending.split(".")
    trending[0], trending[2] = trending[2], trending[0]
    trending = "-".join([x for x in trending])
    trending = parser.parse(trending)

    # cut off timezone
    publish = publish.split('.')[0]
    publish = parser.parse(publish)

    time = trending - publish

    if (time.days <= 1 and int(views) >= 1000000):
        return 1
    return 0


def csv_conversion(file_name):
    df = pd.read_csv(file_name, encoding = "ISO-8859-1")
    df['label'] = df.parallel_apply(lambda x: getLabel(x['publish_time'], x['trending_date'], x['views']), axis=1)
    
    # 39234 rows
    bad = df[df.label == 0]
    # 1715 rows
    good = df[df.label == 1]

    # shuffle and resample(underfit)
    bad.sample(frac=1)
    bad = df.iloc[:good.shape[0],:]

    # combine good and bad
    df = pd.concat([good, bad])

    df['clickbait_title_score'] = df.parallel_apply(lambda x: clickbaitTitle(x['title']), axis=1)
    df['clickbait_image_score'] = df.parallel_apply(lambda x: clickbaitImage(x['thumbnail_link']), axis=1)

    # print average clickbait score
    averageClickbaitScore(df)

    return df

#clickbait # of capital letters in title / total amount of letters
def clickbaitTitle(title):
    return len(re.findall(r'[A-Z]', title)) / len(title)

def averageClickbaitScore(df):
    bad = df[df.label == 0]
    good = df[df.label == 1]

    print("Average good score:")
    print(good['clickbait_title_score'].mean())

    print("Average bad score:")
    print(bad['clickbait_title_score'].mean())



#url = "https://i.ytimg.com/vi/F5mzb086QM8/default.jpg"
#r = 0

# takes in a yt thumbnail url and returns true if there are more than 50 detected letters
# the yt thumbnail (more like a bunch of color/activity in the yt thumbnail)

count= 0

def clickbaitImage(url):
    global count
    try:
        hqres= url.split("/")
        hqres[5] = "hqdefault.jpg"
        hqres= "/".join(str(x) for x in hqres)
        r = requests.get(hqres)
        #print(hqres)
    except requests.exceptions.RequestException as e:
        return 0
    
    img = Image.open(io.BytesIO(r.content))

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img, config = custom_config)

    print(count)
    count= count+1

    if len(re.findall(r'[a-zA-Z]', text)) > 50:
        return 1
    return 0

df_US = csv_conversion("data/USvideos.csv")
df_US.to_csv("data/USProcessed.csv")
#print(averageClickbaitScore(df_US))
#df_RU = csv_conversion("data/RUvideos.csv")
#df_MX = csv_conversion("data/MXvideos.csv")
#df_KR = csv_conversion("data/KRvideos.csv")
#df_JP = csv_conversion("data/JPvideos.csv")
#df_IN = csv_conversion("data/INvideos.csv")
#df_GB = csv_conversion("data/GBvideos.csv")
#df_FR = csv_conversion("data/FRvideos.csv")
#df_DE = csv_conversion("data/DEvideos.csv")
#df_CA = csv_conversion("data/CAvideos.csv")
#df_CA.to_csv("data/CAProcessed.csv")
