import pandas as pd
import numpy as np
import re
from PIL import Image
import requests
import pytesseract
import io
from dateutil import parser
import spacy
import unidecode
from autocorrect import Speller
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split



from pandarallel import pandarallel

os.system("python -m spacy download en")
nltk.download('stopwords')
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
    df = pd.read_csv(file_name, encoding="ISO-8859-1")
    df['label'] = df.parallel_apply(lambda x: getLabel(
        x['publish_time'], x['trending_date'], x['views']), axis=1)

    # 39234 rows
    bad = df[df.label == 0]
    # 1715 rows
    good = df[df.label == 1]

    # shuffle and resample(underfit)
    bad = bad.sample(frac=1)
    bad = df.iloc[:good.shape[0], :]

    # combine good and bad
    df = pd.concat([good, bad])

    # df['clickbait_title_score'] = df.parallel_apply(
    #     lambda x: clickbaitTitle(x['title']), axis=1)
    # df['clickbait_image_score'] = df.parallel_apply(lambda x: clickbaitImage(x['thumbnail_link']), axis=1)

    df['tokenized_title'] = df.parallel_apply(
    lambda x: preprocessor(x['title']), axis=1)

    return df

# clickbait # of capital letters in title / total amount of letters giving punctuation a double count


def clickbaitTitle(title):
    # capital_words = len(re.findall(r'[\b[A-Z]+[0-9]+\b]', title)) * 50
    capitals = len(re.findall(r'[A-Z]', title))
    # question = len(re.findall(r'[?]', title)) * 10
    # exmark = len(re.findall(r'[!]', title)) * 20
    # money = len(re.findall(r'[$]', title)) * 40
    # dash = len(re.findall(r'[\-]', title)) * 10

    return len(title.split(' '))


def averageClickbaitTitleScore(df):
    bad = df[df.label == 0]
    good = df[df.label == 1]

    print("Average good score:")
    print(good['clickbait_title_score'].mean())

    print("Average bad score:")
    print(bad['clickbait_title_score'].mean())


def averageClickbaitImageScore(df):
    bad = df[df.label == 0]
    good = df[df.label == 1]

    print("Average good score:")
    print(good['clickbait_image_score'].mean())

    print("Average bad score:")
    print(bad['clickbait_image_score'].mean())

# takes in a yt thumbnail url and returns true if there are more than 50 detected letters
# the yt thumbnail (more like a bunch of color/activity in the yt thumbnail)


count = 0


def clickbaitImage(url):
    try:
        hqres = url.split("/")
        hqres[5] = "hqdefault.jpg"
        hqres = "/".join(str(x) for x in hqres)
        r = requests.get(hqres)
        print(hqres)
    except requests.exceptions.RequestException as e:
        return 0

    img = Image.open(io.BytesIO(r.content))

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img, config=custom_config)


    if len(re.findall(r'[a-zA-Z]', text)) > 50:
        return 1
    return 0


def preprocessor(review):
    global count

    # remove HTML tags
    nohtml_review = re.sub('<[^<]+?>', '', review)
    # remove unicode characters
    no_unicode = unidecode.unidecode(nohtml_review)
    # convert words to lowercase
    lowercase_review = no_unicode.lower()
    # autocorrect sentence
    speller = Speller()
    autocorrect_review = speller.autocorrect_sentence(lowercase_review)
    # use spacy to lemmatize
    spacy_nlp = spacy.load('en', disable=['parser', 'ner'])
    spacy_output = spacy_nlp(autocorrect_review)
    filtered_review = " ".join([token.lemma_ for token in spacy_output])

    print(count)
    count = count+1
    return filtered_review


# df = pd.read_csv("data/USProcessed.csv", encoding = "ISO-8859-1")
# df_US = csv_conversion("data/USvideos.csv")
# df_US.to_csv("data/test.csv")


df = pd.read_csv("data/test.csv", encoding = "ISO-8859-1")

data = df.drop(columns=['label'])
sentiment = df['label']


x_train, x_test, y_train, y_test = train_test_split(data, sentiment, random_state = 42, test_size = 0.2, shuffle = True)

tfidf = TfidfVectorizer(stop_words=stopwords.words('english'))


train_vector = tfidf.fit_transform(x_train.tokenized_title)
test_vector = tfidf.transform(x_test.tokenized_title)


exit(0)


df_US = csv_conversion("data/USvideos.csv")
df_US.to_csv("data/USProcessed.csv")
# print(averageClickbaitScore(df_US))
#df_RU = csv_conversion("data/RUvideos.csv")
#df_MX = csv_conversion("data/MXvideos.csv")
#df_KR = csv_conversion("data/KRvideos.csv")
#df_JP = csv_conversion("data/JPvideos.csv")
#df_IN = csv_conversion("data/INvideos.csv")
#df_GB = csv_conversion("data/GBvideos.csv")
#df_FR = csv_conversion("data/FRvideos.csv")
#df_DE = csv_conversion("data/DEvideos.csv")
#df_CA = csv_conversion("data/CAvideos.csv")
# df_CA.to_csv("data/CAProcessed.csv")
