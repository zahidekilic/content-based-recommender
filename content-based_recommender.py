#############################
# Content Based Recommendation 
#############################

#############################
# Developing Recommendations Based on Movie Overviews
#############################

# 1. Setting Up TF-IDF Matrix
# 2. Setting Up Cosine Similarity Matrix
# 3. Making Recommendations Based on Similarities

#################################
# 1. Setting Up TF-IDF Matrix
#################################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv", low_memory=False) 
df.head()

df["overview"].head()

#We need to translate textual expressions into measurable, mathematical expressions.
tfidf = TfidfVectorizer(stop_words="english") #we will subtract stop_words because it does not carry any measurement value 

df[df["overview"].isnull()]
df["overview"] = df["overview"].fillna('') 

# transform the overview by calling the vectorizer.
tfidf_matrix = tfidf.fit_transform(df["overview"])

tfidf_matrix.shape 
df["title"].shape 

tfidf.get_feature_names()
tfidf_matrix.toarray() #tf/idf scores

#################################
# 2. Setting Up Cosine Similarity Matrix
#################################

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix) #calculates cosine similarity for all pairs of documents we have one by one.
cosine_sim.shape 

cosine_sim[1] # The similarity score of the 1st index movie with all the other movies

#################################
# 3. Making Recommendations Based on Similarities
#################################
indices = pd.Series(df.index, index = df["title"]) #film names and indexes according to name
indices["Cinderella"]

indices.index.value_counts() #duplicated titles...
indices = indices[~indices.index.duplicated(keep="last")] #We have deduplicated our indexes.
indices.index.value_counts()
indices["Cinderella"]

# Now let's find the similarities from Sherlock Holmes.
indices["Sherlock Holmes"]
movie_index = indices["Sherlock Holmes"]
cosine_sim[movie_index]
similarity_score = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
# top 10 films..
movie_indices = similarity_score.sort_values("score", ascending=False)[1:11].index #0th film is Sherlock Holmes, pass away it.
df['title'].iloc[movie_indices]
