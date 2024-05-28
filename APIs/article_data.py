# Author: Michael Taleb
# email: michael.j.taleb@vanderbilt.edu
# description: API call that returns the article and info about the article
#              but must use article id found in a different API call
#	       youtube tutorial: https://www.youtube.com/watch?v=q-Zgekj-ng0
#	       rapidAPI link: https://rapidapi.com/apidojo/api/seeking-alpha

import requests


# this is a test funciton to get the id of the lastest articles
# this should later be updated to get either articles between a time range or on a certain stock
def get_article_id():
    url = "https://seeking-alpha.p.rapidapi.com/articles/v2/list"

    querystring = {"size": "20", "number": "1", "category": "latest-articles"}

    headers = {
        "X-RapidAPI-Key": "8a28ed0502msh5d73d744c8833f2p1bd458jsn6b9c2f7556ab",
        "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring).json()
    article_id = response['data'][0]['id']
    return article_id


# function that gets API call that gets data
# paramteter id: the id of the article to read gotten from the different API call above
def get_article(id):
    url = "https://seeking-alpha.p.rapidapi.com/articles/get-details"

    querystring = {"id": id}

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


article_id = get_article_id()
date, title, summary, article, author_id = get_article(article_id)

print("date: " + date + "\n")
print("title: " + title + "\n")
print("summary: " + ' '.join(summary) + "\n")
print("text: " + article + "\n")
print("author_id: " + author_id + "\n")
# print()
