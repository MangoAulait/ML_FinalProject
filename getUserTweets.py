"""
Author: Luyao Chen
"""

import tweepy
import csv
import os.path
try:
    import json
except ImportError:
    import simplejson as json


# Import the necessary methods from "twitter" library
# from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream

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
    #Check if the exist
    file_path = r'userTweets\%s_tweets.csv' % screen_name
    if os.path.exists(file_path):
        return
    #Try to download file
    try:
        new_tweets = api.user_timeline(screen_name=screen_name, count=200)

        outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8"), tweet.retweet_count, tweet.retweeted] for
                     tweet in new_tweets]

        # write the csv
        with open(file_path, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "created_at", "text", "retweet_count", "tweet.retweeted"])
            writer.writerows(outtweets)
        pass
    except :
        print screen_name+" tweets content can not be visited"



