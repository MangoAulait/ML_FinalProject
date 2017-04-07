"""
Author: Luyao Chen
"""
import tweepy
import pandas as pd
import csv
try:
    import json
except ImportError:
    import simplejson as json

# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream

# Variables that contains the user credentials to access Twitter API
ACCESS_TOKEN = '838846188271120384-Ae2ZTjblzsuiYXFQWzX8DbAqhZTAfKS'
ACCESS_SECRET = 'UXWHUJHUibYMZJPSHuyWcIfAhsvILuPxcSyQkhKtV80RM'
CONSUMER_KEY = 'nSigahgpmYQcuuU3LqlKhfOhM'
CONSUMER_SECRET = '4Rp9Hd6HuTTTJombqfFAlZpNrvYZ9gmY1iXcaKqpqXPKUSWIkm'


# Initiate the connection to Twitter Streaming API
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)

# Get the newest 200 tweets for a user
def getTweets(screen_name):
    try:
        new_tweets = api.user_timeline(screen_name=screen_name, count=200)

        outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8"), tweet.retweet_count, tweet.retweeted] for
                     tweet in new_tweets]

        # write the csv
        with open(r'userTweets\%s_tweets.csv' % screen_name, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "created_at", "text", "retweet_count", "tweet.retweeted"])
            writer.writerows(outtweets)
        pass
    except :
        print screen_name
#Get the screen name from data
def getAllUsers(location):
    with open(location, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        column = [row[2] for row in reader]
    return column[1:]

bots = getAllUsers(r'C:\Users\dwche\OneDrive\Machine Learning\ML_FinalProject\trainData\bots_data.csv')
for bot in bots:
    getTweets(bot)
nonbots = getAllUsers(r'C:\Users\dwche\OneDrive\Machine Learning\ML_FinalProject\trainData\nonbots_data.csv')
for nonbot in nonbots:
    getTweets(nonbot)