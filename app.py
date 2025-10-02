import streamlit as st
import pandas as pd
import numpy as np
import pickle

scaler = pickle.load(open("scaler.pkl","rb")) 
kmeans = pickle.load(open("kmeans.pkl","rb")) 
user_movie_matrix = pickle.load(open("user_movie_matrix.pkl","rb"))
movies = pickle.load(open("movies.pkl","rb")) 

#Streamlit UI
st.title("Movie Recommendation via Viewer Clustering")
st.write("Select a user to get personalizaed movie recommendations based on similar viewers")

#Select user
user_id = st.selectbox("Choose userId:",user_movie_matrix.index)

#Genrate Recommendations
if st.button("Recommend Movies"):
    user_vector = user_movie_matrix.loc[user_id].values.reshape(1,-1)
    user_vector_scaled = scaler.transform(user_vector)

    cluster_label = kmeans.predict(user_vector_scaled)[0]

    cluster_users = np.where(kmeans.labels_ == cluster_label)[0]
    cluster_users_ids = user_movie_matrix.index[cluster_users]

    cluster_ratings = user_movie_matrix.loc[cluster_users_ids]
    mean_ratings = cluster_ratings.mean().sort_values(ascending=False)

    user_rated = user_movie_matrix.loc[user_id]
    recommendations = mean_ratings[user_rated == 0].head(10).index
    recommended_movies = movies[movies["movieId"].isin(recommendations)]["title"].values

    st.subheader(f"Top 10 recommended movies for user {user_id}")
    for i , movie in  enumerate(recommended_movies , start = 1):
        st.write(f"{i}.{movie}")