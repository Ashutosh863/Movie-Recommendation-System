  --> Movie Recommendation via Viewer Clustering

This project demonstrates a simple **movie recommendation system** using the [MovieLens dataset](https://grouplens.org/datasets/movielens/).  
Instead of traditional collaborative filtering, it clusters users based on their viewing/rating patterns and recommends movies liked by similar viewers.

---

## Features
- Clusters users based on their movie ratings using **KMeans**.
- Uses **PCA** for dimensionality reduction and visualization of user clusters.
- Interactive **Streamlit app** for generating recommendations.
- Provides **top 10 movie recommendations** for any selected user.

---

## Project Structure
.
├── app.py # Streamlit app for recommendations
├── movie_clustering.py # Preprocessing, clustering & model training
├── movies.csv # Movie metadata (MovieLens dataset)
├── ratings.csv # User ratings (MovieLens dataset)
├── scaler.pkl # Saved StandardScaler
├── kmeans.pkl # Saved KMeans model
├── user_movie_matrix.pkl # User-movie ratings matrix
├── movies.pkl # Saved movies metadata
├── clusters.png # Visualization of user clusters
└── README.md # Project documentation

---

## Installation

1. **Clone the repository**:
   git clone <repo llink>
   cd movie-recommendation-clustering
Install dependencies:


pip install -r requirements.txt
Example requirements.txt:

pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
Download the MovieLens dataset (small version recommended for quick testing: ml-latest-small)
Place movies.csv and ratings.csv in the project directory.

Usage
1. Train Clustering Model
Run the preprocessing & clustering script:


python movie_clustering.py
This will:
Build the user–movie rating matrix
Scale data
Perform PCA & KMeans clustering
Save models (scaler.pkl, kmeans.pkl, user_movie_matrix.pkl, movies.pkl)

Generate a cluster visualization (clusters.png)

2. Launch Streamlit App
Run the interactive app:

streamlit run app.py

How It Works
Select a User ID from the dropdown.

The system:
Scales the user’s ratings
Predicts their cluster with KMeans
Finds similar users in the same cluster
Aggregates ratings from those users
Recommends the top 10 movies the selected user hasn’t rated yet
The recommendations are displayed on the app interface.

📊 Example Output
Cluster Visualization
Streamlit App
When you select a user, the app outputs recommendations like:

Top 10 Recommended Movies for User 12:
1. Toy Story (1995)
2. Shawshank Redemption, The (1994)
3. Pulp Fiction (1994)
...
Tech Stack
Python (pandas, numpy, scikit-learn)
Machine Learning: KMeans, PCA

Visualization: matplotlib, seaborn

Web App: Streamlit

📌 Notes
This is a basic recommendation system — it doesn’t yet use advanced collaborative filtering or deep learning.

For larger datasets, you may want to optimize memory usage (e.g., using sparse matrices).
