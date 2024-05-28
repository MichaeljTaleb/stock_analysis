# Author: Michael Taleb
# email: michael.j.taleb@vanderbilt.edu
# description: API call that returns the article and info about the article
#              but must use article id found in a different API call
#	       youtube tutorial: https://www.youtube.com/watch?v=q-Zgekj-ng0
#	       rapidAPI link: https://rapidapi.com/apidojo/api/seeking-alpha

import requests

# functions to check in a caseinsensitive manner whether or not the title
# mentions the stock
def contains_case_insensitive(title, stock):
    return stock.lower() in title.lower()

# Gets the ids of all the articles in a specified time range
# Parameter is the name of the stock to look for
def get_article_id(stock):
    url = "https://seeking-alpha.p.rapidapi.com/articles/v2/list"

    # since and until is the date range that the articles are fetched from
    querystring = {"until": "1707986312", "since": "1676450312", "size": "100", "number": "1",
                   "category": "latest-articles"}

    headers = {
        "X-RapidAPI-Key": "8a28ed0502msh5d73d744c8833f2p1bd458jsn6b9c2f7556ab",
        "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring).json()
    article_ids = []
    for article in response["data"]:
        if contains_case_insensitive(article["attributes"]["title"], stock):
            article_ids.append(article["id"])

    return article_ids


# function that gets API call that gets data
# paramteter id: the id of the article to read gotten from the different API call above
def get_article(id):
    url = "https://seeking-alpha.p.rapidapi.com/articles/get-details"

    querystring = {"id": id}

    headers = {
        "X-RapidAPI-Key": "8a28ed0502msh5d73d744c8833f2p1bd458jsn6b9c2f7556ab",
        "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)


    # Check if the response is JSON
    # ChatGPT wrote this code to exception check since not all of the API calls were in JSON format
    if response.headers.get('Content-Type') == 'application/json':
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError as e:
            print("JSONDecodeError:", e)
            print("Response content:", response.text)
            return None

        # Extract data to print/return
        try:
            attributes = data['data']['attributes']
            date = attributes['publishOn']
            title = attributes['title']
            summary = attributes['summary']
            article = attributes['content']
            author_id = data['data']['relationships']['author']['data']['id']
            return date, title, summary, article, author_id
        except KeyError as e:
            print("KeyError:", e)
            print("Response content:", response.text)
            return None
    else:
        print("Unexpected response format:", response.text)
        return None

    # if no errors were encountered just return the fields we need
    return date, title, summary, article, author_id


article_ids = get_article_id("gold")

for article_id in article_ids:
    date, title, summary, article, author_id = get_article(article_id)

    print("date: " + date + "\n")
    print("title: " + title + "\n")
    print("summary: " + ' '.join(summary) + "\n")
    # print("text: " + article + "\n")
    print("author_id: " + author_id + "\n")
    # print()
