from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pandas as pd
import re
from PIL import Image
import requests
import pytesseract
import io
import spacy
import unidecode
from autocorrect import Speller
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from utils import *


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


def csv_conversion():
    df1 = pd.read_csv("data/USvideos.csv", encoding="ISO-8859-1")
    df2 = pd.read_csv("data/CAvideos.csv", encoding="ISO-8859-1")
    df3 = pd.read_csv("data/GBvideos.csv", encoding="ISO-8859-1")
    df4 = pd.read_csv("data/DEvideos.csv", encoding="ISO-8859-1")

    df = pd.concat([df1, df2, df3, df4])

    df['label'] = df.apply(lambda x: getLabel(
        x['publish_time'], x['trending_date'], x['views']), axis=1)

    bad = df[df.label == 0]
    good = df[df.label == 1]

    # shuffle and resample(underfit)
    bad = bad.sample(frac=1)
    bad = df.iloc[:good.shape[0], :]

    # combine good and bad
    df = pd.concat([good, bad])

    # clickbait title and image score

    # df['clickbait_title_score'] = df.apply(
    #     lambda x: clickbaitTitle(x['title']), axis=1)
    # df['clickbait_image_score'] = df.apply(lambda x: clickbaitImage(x['thumbnail_link']), axis=1)

    # preprocess title, description, tags

    # df['tokenized_title'] = df.apply(
    # lambda x: preprocessor(x['title']), axis=1)
    # df['tokenized_tags'] = df.apply(
    # lambda x: preprocessor(x['tags']), axis=1)
    # df['tokenized_description'] = df.apply(
    # lambda x: preprocessor(x['description']), axis=1)

    df['title_sent_class'] = df.apply(
        lambda x: classifiedTextblob(x['title']), axis=1)
    df['tags_sent_class'] = df.apply(
        lambda x: classifiedTextblob(x['tags']), axis=1)
    df['descrip_sent_class'] = df.apply(lambda x: classifiedTextblob(str(x['description'])), axis=1)

    df['time_of_day'] = df.apply(lambda x: getTimeOfDay(
        x['publish_time']), axis=1)

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

# conversion

# df = csv_conversion()
# df.to_csv("data/USCAGBDEProcessedTextBlob.csv")


# integrity check

# df = pd.read_csv("data/USCAGBDEProcessedTextBlob.csv", encoding="ISO-8859-1")
# bad = df[df.label == 0]
# good = df[df.label == 1]

# print(bad)
# print(good)



exit(0)


# compute("test", "test", ['test', 'test'], 0, 32)

# FEATURE SELECTION
df = pd.read_csv("data/USCAGBDEProcessedTextBlob.csv", encoding="ISO-8859-1")
data = df[['category_id', 'title_sent_class', 'tags_sent_class',
           'descrip_sent_class', 'time_of_day', 'label']]

data = data.dropna()
data.to_csv("data/test.csv")


sentiment = data['label']
data = data.drop(columns=['label'])


x_train, x_test, y_train, y_test = train_test_split(
    data, sentiment, random_state=42, test_size=0.2, shuffle=True)

target_names = ['class 0', 'class 1']

# Code that runs the K-Nearest Neighbors model.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(classification_report(y_test, y_pred, target_names=target_names))

# from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor(random_state = 42)
# from pprint import pprint
# from sklearn.model_selection import RandomizedSearchCV
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# max_features = ['sqrt', 'log2', None]
# max_depth = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
# max_depth.append(None)
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# bootstrap = [True, False]
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf,
#               'bootstrap': bootstrap}

# pprint(random_grid)
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 500, cv = 5, verbose = 2, n_jobs = -1)
# rf_random.fit(x_train, y_train)
# print(rf_random.best_params_)

# n_estimators= 600, min_samples_split= 5, min_samples_leaf = 1, max_features = 'log2', max_depth = 40, bootstrap = False)
random_forest = RandomForestClassifier()
random_forest = random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
#print(classification_report(y_test, y_pred, target_names = target_names))

decision_tree = DecisionTreeClassifier()
decision_tree = decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
print(classification_report(y_test, y_pred, target_names=target_names))

# fig = plt.figure(figsize = (25, 20))
# _ = tree.plot_tree(decision_tree,
#                    feature_names = list(data.columns),
#                    class_names = target_names,
#                    filled = True)
#plt.show()

# from sklearn.neural_network import MLPClassifier
# neural_network = MLPClassifier(hidden_layer_sizes = (100, 100, 100), max_iter = 1000)
# neural_network = neural_network.fit(x_train, y_train.values.ravel())
# y_pred = neural_network.predict(x_test)
#print(classification_report(y_test, y_pred, target_names = target_names))



