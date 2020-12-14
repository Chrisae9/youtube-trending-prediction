from textblob import TextBlob
from dateutil import parser
import random
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def classifiedTextblob(data):
    polar = TextBlob(data).sentiment.polarity
    if (polar < 0):
        return -1
    if (polar == 0):
        return 0
    if (polar > 0):
        return 1


def getTimeOfDay(publish):

    # cut off timezone
    publish = publish.split('.')[0]
    publish = parser.parse(publish)

    time = publish.hour

    # night
    if time <= 6:
        return 3
    # morning
    if time <= 12:
        return 0
    # afternoon
    if time <= 18:
        return 1
    # evening
    if time <= 24:
        return 2
    return random.randint(0, 3)


def compute(title, description, tags, time_of_day, category):
    df = pd.read_csv("data/USCAGBDEProcessedTextBlob.csv",
                     encoding="ISO-8859-1")
    data = df[['category_id', 'title_sent_class', 'tags_sent_class',
               'descrip_sent_class', 'time_of_day', 'label']]

    data = data.dropna()

    sentiment = data['label']
    data = data.drop(columns=['label'])

    class_title = classifiedTextblob(title)
    class_description = classifiedTextblob(description)
    class_tags = classifiedTextblob(('""|""').join(tags))

    d = {'title_sent_class': class_title, 'tags_sent_class': class_tags,
         'descrip_sent_class': class_description, 'time_of_day': time_of_day, 'category_id': category}
    input = pd.DataFrame(data=d, index=[0])

    decision_tree = DecisionTreeClassifier()
    decision_tree = decision_tree.fit(data, sentiment)
    prediction = decision_tree.predict(input)

    return prediction[0]
