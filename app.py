import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

df = pd.read_csv('data/books.csv')
df.columns = df.columns.str.strip()

if 'rating_obj' not in df.columns:
    print("Column 'rating_obj' not found in the dataset.")
    rating_df = pd.DataFrame()
else:
    rating_df = pd.get_dummies(df['rating_obj'])

if 'categories' not in df.columns:
    print("Column 'categories' not found in the dataset.")
    language_df = pd.DataFrame()
else:
    language_df = pd.get_dummies(df['categories'])

df.drop(['id', 'num_pages', 'ratings_count', 'subtitle', 'isbn10'], axis=1, inplace=True)

features = pd.concat([rating_df, language_df, df[['average_rating', 'published_year']]], axis=1)
features.set_index(df['title'], inplace=True)

imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_imputed)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['description'].fillna(''))

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

model = neighbors.NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
model.fit(tfidf_matrix)

@app.route('/')
def index():
    return render_template('index.html', books=df['title'].unique())

@app.route('/recommend', methods=['POST'])
def recommend():
    search_text = request.form['search_text']
    search_vector = vectorizer.transform([search_text])

    if not re.search(r'\w', search_text):
        return render_template('recommendations.html', books=[], message='Please enter valid search words.')

    distances, indices = model.kneighbors(search_vector, n_neighbors=4)

    recommended_books = []
    graph_paths = []

    for i in range(len(indices.flatten())):
        description = df.iloc[indices.flatten()[i]]['description']
        sentiment = TextBlob(description).sentiment
        polarity_category = 'Positive' if sentiment.polarity > 0 else 'Negative' if sentiment.polarity < -0 else 'Neutral'

        # Generate bar graph for sentiment analysis
        plt.figure(figsize=(6, 4))
        plt.bar(['Positive', 'Neutral', 'Negative'],
                [1 if sentiment.polarity > 0 else 0, 1 if -0 <= sentiment.polarity <= 0 else 0, 1 if sentiment.polarity < -0 else 0],
                color=['green', 'gray', 'red'])
        plt.ylim([0, 1])
        plt.title(f"Sentiment Analysis for '{df.iloc[indices.flatten()[i]]['title']}'")
        img_path = f"static/{df.iloc[indices.flatten()[i]]['title'].replace(' ', '_')}_sentiment.png"
        plt.savefig(img_path)
        plt.close()

        graph_paths.append(img_path)

        book_info = {
            'title': df.iloc[indices.flatten()[i]]['title'],
            'author': df.iloc[indices.flatten()[i]]['authors'] if 'authors' in df.columns else 'Unknown',
            'thumbnail': df.iloc[indices.flatten()[i]]['thumbnail'] if 'thumbnail' in df.columns else 'No image available',
            'description': description,
            'polarity': sentiment.polarity,
            'polarity_category': polarity_category,
            'sentiment_graph': img_path
        }
        recommended_books.append(book_info)

    return render_template('recommendations.html', books=recommended_books, graph_paths=graph_paths)

@app.route('/sentiment_analysis')
def sentiment_analysis():
    graph_paths = request.args.getlist('graph_paths')
    return render_template('sentiment_analysis.html', graph_paths=graph_paths)

if __name__ == '__main__':
    # Ensure the static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
