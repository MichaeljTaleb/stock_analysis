# Author: Michael Taleb
# email: michael.j.taleb@vanderbilt.edu
# description: API call that returns the article and info about the article
#              but must use article id found in a different API call
#			   youtube tutorial: https://www.youtube.com/watch?v=q-Zgekj-ng0

import requests

# function that gets API call that gets data
# paramteter id: the id of the article to read gotten from a different API call
def get_article(id):
	url = "https://seeking-alpha.p.rapidapi.com/articles/get-details"

	querystring = {"id":id}

	headers = {
		"X-RapidAPI-Key": "8a28ed0502msh5d73d744c8833f2p1bd458jsn6b9c2f7556ab",
		"X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
	}


	response = requests.get(url, headers=headers, params=querystring).json()

# 	data to print/return
	attributes = response['data']['attributes']
	date = attributes['publishOn']
	title = attributes['title']
	summary = attributes['summary']
	article = attributes['content']
	author_id = response['data']['relationships']['author']['data']['id']
	return date, title, summary, article, author_id


date, title, summary, article, author_id = get_article()

print(date)
print(title)
print(summary)
print(article)
print(author_id + "\n")
# print()
