import numpy as np
import pandas as pd
import ast

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# print(credits['crew'])
movies= movies.merge(credits,on='title')
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# print(movies.isnull().sum())
# print(movies.duplicated().sum())
# print(movies.iloc[0].genres)

import ast

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 4:
            L.append(i['name'])
        counter+=1
    return L
movies['cast'] = movies['cast'].apply(convert3)

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L
movies['crew']=movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df=movies[['movie_id','title','tags']]

new_df = movies[['movie_id', 'title', 'tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(new_df['tags']).toarray()


print(vector)
