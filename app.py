import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load dataset
movies = pd.read_csv("movies.csv")

# keep useful columns
movies = movies[['movieId', 'title', 'genres']]

# handle missing values
movies['genres'] = movies['genres'].fillna('')

# convert text into vectors
vectorizer = TfidfVectorizer(stop_words='english')
genre_matrix = vectorizer.fit_transform(movies['genres'])

# compute similarity
similarity = cosine_similarity(genre_matrix)

# recommendation function
def recommend(movie_name):
    if movie_name not in movies['title'].values:
        print("Movie not found")
        return

    idx = movies[movies['title'] == movie_name].index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_5 = scores[1:6]

    print("\nRecommendations for:", movie_name)
    for i in top_5:
        print("-", movies.iloc[i[0]].title)

# test
recommend("Toy Story (1995)")
