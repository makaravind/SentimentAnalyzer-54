import tweepy
from textblob import TextBlob

consumer_key = 'KGgCMkLGg8F0i4cJCbi0ed2uK'
consumer_secret = 'CASj9IzsoJIF5DmGsD12SOAOK8e7o7QEScJiwIbKeMiAN81Hgd'

access_token = '1707106273-rhLapzgoGez0XQSVKIEgSEtKRQCrQtX9HA5kUOi'
access_token_secret = 'jfTTQl66h76o812miVUKCBV5h72tPFVq5aKiEpSD0BN4w'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweet = api.search('black money')

for tweet in public_tweet:
    print tweet.text
    analysis = TextBlob(tweet.text)
    print analysis.sentiment