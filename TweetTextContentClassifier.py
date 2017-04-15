"""
Author: Luyao Chen
"""


import os.path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier

# Retrive data
path_to_nonbot = 'C:\Users\Luyao Chen\OneDrive\Machine Learning\ML_FinalProject\\trainData\\'
path_to_bot = 'C:\Users\Luyao Chen\OneDrive\Machine Learning\ML_FinalProject\\trainData\\'
path_to_userTweets = 'C:\Users\Luyao Chen\OneDrive\Machine Learning\ML_FinalProject\\userTweets\\'

data_nonbot = pd.read_csv(path_to_nonbot + 'nonbots_data.csv', header = 0)
data_bot = pd.read_csv(path_to_bot + 'bots_data.csv', header = 0)
data = pd.concat([data_nonbot, data_bot])

gathered_tweets = ['']
for name in data['screen_name']:
    # Remove double quotes
    if name.startswith('"') and name.endswith('"'):
        name = name[1:-1]
    # Check whether the file exists
    gathered_tweet = ''
    if os.path.isfile(path_to_userTweets + name + '_tweets.csv'):
        temp_data = pd.read_csv(path_to_userTweets + name + '_tweets.csv', header = 0)
        # Gather tweets
        for single_tweet in temp_data['text']:
            gathered_tweet = gathered_tweet + str(single_tweet)
    gathered_tweets.append(gathered_tweet)

# Build the train and test data
X_train, X_test, y_train, y_test = train_test_split(np.array(gathered_tweets[1:]), np.array(data['bot']), test_size=0.25, random_state=0)

# MultinomialNB
clf1 = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('clf', MultinomialNB())])
clf1 = clf1.fit(X_train, y_train)

# Metrics & Auc for MultinomialNB
predicted = clf1.predict(X_test)
auc_score1 = roc_auc_score(y_test, predicted)
print "MultinomialNB Classification Report"
print metrics.classification_report(y_test, predicted,
     target_names=['0','1'])
print 'Auc score: ' + str(auc_score1)


# BernoulliNB
clf2 = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('clf', BernoulliNB())])
clf2 = clf2.fit(X_train, y_train)

# Metrics & Auc for BernoulliNB
predicted = clf2.predict(X_test)
auc_score2 = roc_auc_score(y_test, predicted)
print "BernoulliNB Classification Report"
print metrics.classification_report(y_test, predicted,
                                    target_names=['0','1'])
print 'Auc score: ' + str(auc_score2)

# Random Forest
clf3 = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('clf', RandomForestClassifier())])
clf3 = clf3.fit(X_train, y_train)

# Metrics & Auc for Random Forest
predicted = clf3.predict(X_test)
auc_score3 = roc_auc_score(y_test, predicted)
print "RandomForestClassifier Classification Report"
print metrics.classification_report(y_test, predicted,
                                    target_names=['0','1'])
print 'Auc score: ' + str(auc_score3)


