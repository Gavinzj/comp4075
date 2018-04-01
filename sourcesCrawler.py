import requests
import json
import io
import sys

names = ['abc-news', 'abc-news-au', 'aftenposten','al-jazeera-english','ars-technica','associated-press','australian-financial-review','axios', 'bbc-news', 'bbc-sport','bleacher-report', 'bloomberg','breitbart-news','business-insider', 'business-insider-uk','buzzfeed','cbc-news', 'cbs-news','cnbc','cnn','crypto-coins-news','daily-mail','engadget','entertainment-weekly','espn','engadget','espn-cric-info','financial-post','financial-times','football-italia','fortune','fox-sports','fox-news','four-four-two','google-news','google-news-ca','google-news-uk','google-news-in''google-news-au','hacker-new','ign','independent','mashable','metro','mirror','mtv-news','medical-news-today','mtv-news-uk','national-geographic','msnbc','nbc-news','news24','new-scientist','newsweek','news-com-au','new-york-magazine','next-big-future','nfl-news','nhl-news','politico','polygon','recode','reuters','reddit-r-all','rte','techradar','the-economist','the-globe-and-mail','the-guardian-au','the-guardian-uk','techcrunch','the-hill','talksport','the-hindu','the-irish-times','the-lad-bible','the-huffington-post','the-new-york-times','the-times-of-india','the-telegraph','the-verge','the-wall-street-journal','the-washington-post','time','usa-today','vice-news','wired','xinhua-net','der-tagesspiegel']
sys.stdout=open("/sources/output20180401.json","a+")
print("[")
sys.stdout.close()
for name in names:
	url = ('https://newsapi.org/v2/everything?sources='+name+'&pageSize=100&language=en&from=2018-04-01&to=2018-04-01&apiKey=c0456841cb6a4dc794e3ec64e86b7e6e')
	count = 0
	response = requests.get(url) 
	sys.stdout=open("/sources/output20180401.json","a+")
	print(json.dumps(response.json()))
	print(",")
	sys.stdout.close()

sys.stdout=open("/sources/output20180401.json","a+")
print("]")
sys.stdout.close()
#&from=2018-03-28
