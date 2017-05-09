"""
Author: Luyao Chen & Qijie Huang
"""

import pandas as pd
import numpy as np

import dataProcess
import getUserTweets
import TweetTextContentClassifier
import TweetTimeSeriesClassifer
import TweetNumericClassifier
import TweetNonNumericClassifier

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import training data to Pandas DataFrame
path_to_nonbot = 'C:\Users\dwche\OneDrive\Machine Learning\ML_FinalProject\\traindata\\'
path_to_bot = 'C:\Users\dwche\OneDrive\Machine Learning\ML_FinalProject\\traindata\\'
path_to_userTweets = 'C:\Users\dwche\OneDrive\Machine Learning\ML_FinalProject\\userTweets\\'

# Retrive data
data_nonbot = pd.read_csv(path_to_nonbot + 'nonbots_data.csv', header = 0, encoding='latin-1')
data_bot = pd.read_csv(path_to_bot + 'bots_data.csv', header = 0, encoding='latin-1')
data = pd.concat([data_nonbot, data_bot])

# Clean the data
df = dataProcess.clean_data_process(data)
df = dataProcess.fill_Na(data)



'''
# Gather the Users' tweet content and Store the data in
df["screen_name"].apply(getUserTweets.getTweets)
'''




#Classify the tweet count based mutiple classifier result
X_train_hidden, X_temp, y_train_hidden, y_temp = train_test_split(np.array(df[df.columns[:19]]), np.array(df['bot']), test_size=0.4, random_state=0)
X_test_hidden, X_test, y_test_hidden, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=0)

# Train hidden classifier
l1 = TweetTimeSeriesClassifer.TimeSeriesClassify(X_train_hidden, X_test_hidden, y_train_hidden, y_test_hidden)
l2 = TweetTextContentClassifier.TweetContentClassify(X_train_hidden, X_test_hidden, y_train_hidden, y_test_hidden)
l3 = TweetNumericClassifier.TweetNumericClassify(X_train_hidden, X_test_hidden, y_train_hidden, y_test_hidden)
l4 = TweetNonNumericClassifier.TweetNonNumericClassify(X_train_hidden, X_test_hidden, y_train_hidden, y_test_hidden)

# Constract the bot training set
l = [l1] + [l2] + [l3] + l4
# l = [l3] + l4

X_train_bot = np.asarray(l).T.tolist()

# Train bot classifier
clf = RandomForestClassifier()
clf = clf.fit(X_train_bot , y_test_hidden)

# Get test set for bot classifier
l5 = TweetTimeSeriesClassifer.TimeSeriesClassify(X_train_hidden, X_test, y_train_hidden, y_test)
l6 = TweetTextContentClassifier.TweetContentClassify(X_train_hidden, X_test, y_train_hidden, y_test)
l7 = TweetNumericClassifier.TweetNumericClassify(X_train_hidden, X_test, y_train_hidden, y_test)
l8 = TweetNonNumericClassifier.TweetNonNumericClassify(X_train_hidden, X_test, y_train_hidden, y_test)

# Constract the bot testing set
l_ = [l5] + [l6] + [l7] + l8
# l_ = [l7] + l8
x_test_bot = np.asarray(l_).T.tolist()

# Make prediction
pred = clf.predict(x_test_bot)

print 'Accuracy_no_hard-coded_rule: ' + str(accuracy_score(y_test, pred))

# add some hard-coded rule
for index, name in enumerate(X_test.T[2].tolist()):
    if 'bot' in name.lower():
        pred[index] = 1

print 'Accuracy_with_hard-coded_rule: ' + str(accuracy_score(y_test, pred))

















