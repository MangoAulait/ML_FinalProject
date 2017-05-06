"""
Author: Luyao Chen
"""

import pandas as pd
import numpy as np

import dataProcess
import getUserTweets
import TweetTextContentClassifier
import TweetTimeSeriesClassifer

#Import training data to Pandas DataFrame
path_to_nonbot = 'C:\Users\Luyao Chen\OneDrive\Machine Learning\ML_FinalProject\\trainData\\'
path_to_bot = 'C:\Users\Luyao Chen\OneDrive\Machine Learning\ML_FinalProject\\trainData\\'

data_nonbot = pd.read_csv(path_to_nonbot + 'nonbots_data.csv', header = 0)
data_bot = pd.read_csv(path_to_bot + 'bots_data.csv', header = 0)

data = pd.concat([data_nonbot, data_bot])


#Clean the data
df = dataProcess.clean_data_process(data)

'''
#Gather the Users' tweet content and Store the data in
df["screen_name"].apply(getUserTweets.getTweets)
'''


#TweetTimeSeriesClassification
l1 = TweetTimeSeriesClassifer.TimeSeriesClassify(df)

#TweetContentClassification
l2 = TweetTextContentClassifier.TweetContentClassify(df)


