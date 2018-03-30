from twitter_scraper import get_tweets

for tweet in get_tweets('TeamCoco', pages=2):
	print("\n\n *** TWEET ***")
	print("\n")
	print(tweet['text'])
	print("\n")
	print("**** THE END ****\n\n")
