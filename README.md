# Is it Possible to Tell if a YouTube Video Will Go “Super Trending”?

## Definition

“Super Trending” - A Trending YouTube Video that obtains over a million views within a 24-hour time period.

## Description

Given recent times with the ongoing pandemic, content creators on YouTube may notice a gain of a larger audience as people look
for things to do within their free time. To take advantage of this, analyzing what exactly makes a video trend is important to
creators when seeking out ways to increase views. This project attempts to predict the performance of videos such that they would
go trending within a day of release. An equation was developed manually to separate videos into two classes: those that reach a
million views or more within twenty-four hours, and those that do not. The category, title, tags, description, and the publish date
were selected as features for the algorithm. Multiple machine learning algorithms were used to predict determine which gave the
best performance. In doing so, Random Forest and the Decision Tree classifiers ended up giving the best results.


## Setup

Install Dependencies.

`pip install -r requirements`

## Run

Launch the GUI.

`python yt.py`

## Resources

### Data

https://www.kaggle.com/datasnaek/youtube-new

### Video

https://www.youtube.com/watch?v=2rJ86XD_bCw&feature=youtu.be
