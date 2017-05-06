"""
Author: Luyao Chen
"""
import os.path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn import tree
from sklearn import linear_model
from sklearn.svm import SVC

# Retrive data
path_to_userTweets = 'C:\Users\Luyao Chen\OneDrive\Machine Learning\ML_FinalProject\\userTweets\\'

#Get Information from local csv files
def get_TweetSeries_Data(names):

    allTimeLines = []

    for name in names:
        singleTimeLine = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if os.path.isfile(path_to_userTweets + name + '_tweets.csv'):
            temp_data = pd.read_csv(path_to_userTweets + name + '_tweets.csv', header=0)
            # Gather tweets
            for tweet_time in temp_data['created_at']:
                hour = tweet_time[11:13]
                singleTimeLine[int(hour)] += 1

        allTimeLines.append(singleTimeLine)

    normalized_timelines = normalize(np.array(allTimeLines))

    var_of_normalized_timelines=np.apply_along_axis(get_var,1,normalized_timelines)

    return normalized_timelines

def get_var(row):
    value = [np.var(row)]
    return value

def listOfList_to_list2(listOfList):
    new_list = ['']
    for l in listOfList.tolist():
        new_list.append(l[1])
    return new_list

def TimeSeriesClassify(data):

    timedata = get_TweetSeries_Data(data["screen_name"])

    X_train, X_test, y_train, y_test = train_test_split(timedata, np.array(data['bot']), test_size=0.25, random_state=0)

    clf3 = Pipeline([('clf', RandomForestClassifier())])
    clf3 = clf3.fit(X_train, y_train)
    predicted_proba = clf3.predict_proba(X_test)
    return listOfList_to_list2(predicted_proba)

    # Metrics & Auc for Random Forest
    # predicted = clf3.predict(X_test)
    # auc_score3 = roc_auc_score(y_test, predicted)
    # print predicted
    # print "RandomForestClassifier Classification Report"
    # print metrics.classification_report(y_test, predicted,
    #                                     target_names=['0','1'])
    # print 'Auc score: ' + str(auc_score3)


    # # MultinomialNB
    # clf1 = Pipeline([('clf', MultinomialNB())])
    # clf1 = clf1.fit(X_train, y_train)
    #
    # # Metrics & Auc for MultinomialNB
    # predicted = clf1.predict(X_test)
    # auc_score1 = roc_auc_score(y_test, predicted)
    # print "MultinomialNB Classification Report"
    # print metrics.classification_report(y_test, predicted,
    #                                     target_names=['0', '1'])
    # print 'Auc score: ' + str(auc_score1)
    #
    # # BernoulliNB
    # clf2 = Pipeline([('clf', BernoulliNB())])
    # clf2 = clf2.fit(X_train, y_train)
    #
    # # Metrics & Auc for BernoulliNB
    # predicted = clf2.predict(X_test)
    # auc_score2 = roc_auc_score(y_test, predicted)
    # print "BernoulliNB Classification Report"
    # print metrics.classification_report(y_test, predicted,
    #                                     target_names=['0', '1'])
    # print 'Auc score: ' + str(auc_score2)
    #
    # # logistic Regression
    # clf4 = linear_model.LogisticRegression()
    # clf4 = clf4.fit(X_train, y_train)
    #
    # # Metrics & Auc Regression
    # predicted = clf4.predict(X_test)
    # auc_score4 = roc_auc_score(y_test, predicted)
    # print "logistic Regression Classification Report"
    # print metrics.classification_report(y_test, predicted,
    #                                     target_names=['0', '1'])
    # print 'Auc score: ' + str(auc_score4)
    #
    # # logistic Regression
    # clf5 = SVC()
    # clf5 = clf5.fit(X_train, y_train)
    #
    # # Metrics & Auc Regression
    # predicted = clf5.predict(X_test)
    # auc_score5 = roc_auc_score(y_test, predicted)
    # print "SVM Classification Report"
    # print metrics.classification_report(y_test, predicted,
    #                                     target_names=['0', '1'])
    # print 'Auc score: ' + str(auc_score5)