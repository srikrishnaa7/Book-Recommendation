import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'model_cache')

POPULAR_PKL = os.path.join(CACHE_DIR, 'popular.pkl')
PT_PKL = os.path.join(CACHE_DIR, 'pt.pkl')
BOOKS_PKL = os.path.join(CACHE_DIR, 'books.pkl')
SIM_PKL = os.path.join(CACHE_DIR, 'similarity_scores.pkl')
TFIDF_VEC_PKL = os.path.join(CACHE_DIR, 'tfidf_vec.pkl')
TFIDF_MAT_PKL = os.path.join(CACHE_DIR, 'tfidf_mat.pkl')
TITLES_PKL = os.path.join(CACHE_DIR, 'titles.pkl')

class BookRecommender:
    def __init__(self):
        self.popular_df = None
        self.pt = None
        self.books_df = None
        self.similarity_scores = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.all_titles = []
        self._load_or_build_models()

    def _load_or_build_models(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Check if cache exists
        cache_files = [POPULAR_PKL, PT_PKL, BOOKS_PKL, SIM_PKL, TFIDF_VEC_PKL, TFIDF_MAT_PKL, TITLES_PKL]
        all_cached = all(os.path.exists(f) for f in cache_files)

        if all_cached:
            logger.info("[Recommender] Loading models from cache...")
            with open(POPULAR_PKL, 'rb') as f:
                self.popular_df = pickle.load(f)
            with open(PT_PKL, 'rb') as f:
                self.pt = pickle.load(f)
            with open(BOOKS_PKL, 'rb') as f:
                self.books_df = pickle.load(f)
            with open(SIM_PKL, 'rb') as f:
                self.similarity_scores = pickle.load(f)
            with open(TFIDF_VEC_PKL, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            with open(TFIDF_MAT_PKL, 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            with open(TITLES_PKL, 'rb') as f:
                self.all_titles = pickle.load(f)
            logger.info(f"[Recommender] Cache loaded successfully. Available popular books: {len(self.popular_df)}")
        else:
            logger.info("[Recommender] Cache missing. Building recommendation models from CSVs...")
            self.build_models()

    def build_models(self):
        books_path = os.path.join(DATA_DIR, 'Books.csv')
        ratings_path = os.path.join(DATA_DIR, 'Ratings.csv')
        
        if not os.path.exists(books_path) or not os.path.exists(ratings_path):
            raise FileNotFoundError(f"Missing dataset files in {DATA_DIR}")

        books = pd.read_csv(books_path, low_memory=False)
        ratings = pd.read_csv(ratings_path)

        # Clean column names
        books.columns = books.columns.str.strip()
        ratings.columns = ratings.columns.str.strip()

        # Merge ratings with books
        ratings_with_name = ratings.merge(books, on='ISBN')

        # 1. Popularity-Based Model (> 250 ratings)
        num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
        num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

        avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
        avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

        popular = num_rating_df.merge(avg_rating_df, on='Book-Title')
        popular = popular[popular['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)

        popular_df = popular.merge(books, on='Book-Title').drop_duplicates('Book-Title')[
            ['Book-Title', 'Book-Author', 'Image-URL-M', 'Image-URL-L', 'num_ratings', 'avg_rating', 'Year-Of-Publication', 'Publisher']
        ]
        popular_df['avg_rating'] = popular_df['avg_rating'].round(2)
        self.popular_df = popular_df

        # 2. Collaborative Filtering Model
        # Active users (>200 votes)
        user_counts = ratings_with_name.groupby('User-ID').count()['Book-Rating']
        active_users = user_counts[user_counts > 200].index
        filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(active_users)]

        # Popular books among active users (>=50 votes)
        book_counts = filtered_rating.groupby('Book-Title').count()['Book-Rating']
        famous_books = book_counts[book_counts >= 50].index

        final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
        pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
        pt.fillna(0, inplace=True)
        self.pt = pt

        from sklearn.metrics.pairwise import cosine_similarity
        self.similarity_scores = cosine_similarity(pt)

        # 3. Clean Books DF for Metadata Lookups
        books_clean = books.drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-M', 'Image-URL-L']].copy()
        self.books_df = books_clean

        # 4. Content-Based TF-IDF Model for Title/Author Search
        books_clean['search_content'] = (books_clean['Book-Title'].fillna('') + ' ' + books_clean['Book-Author'].fillna('') + ' ' + books_clean['Publisher'].fillna('')).astype(str)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(books_clean['search_content'])

        self.all_titles = books_clean['Book-Title'].dropna().unique().tolist()

        # Save to cache
        with open(POPULAR_PKL, 'wb') as f:
            pickle.dump(self.popular_df, f)
        with open(PT_PKL, 'wb') as f:
            pickle.dump(self.pt, f)
        with open(BOOKS_PKL, 'wb') as f:
            pickle.dump(self.books_df, f)
        with open(SIM_PKL, 'wb') as f:
            pickle.dump(self.similarity_scores, f)
        with open(TFIDF_VEC_PKL, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        with open(TFIDF_MAT_PKL, 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)
        with open(TITLES_PKL, 'wb') as f:
            pickle.dump(self.all_titles, f)

        logger.info("[Recommender] Models built and cached successfully!")

    def get_popular_books(self, top_n=50):
        records = self.popular_df.head(top_n).to_dict(orient='records')
        results = []
        for r in records:
            desc = r.get('description') or r.get('Description') if ('description' in r or 'Description' in r) else None
            img = r.get('Image-URL-L') or r.get('Image-URL-M') or '/static/images/no-cover.jpg'
            results.append({
                'title': r['Book-Title'],
                'author': r['Book-Author'],
                'image': img,
                'num_ratings': int(r['num_ratings']),
                'avg_rating': float(r['avg_rating']),
                'year': r.get('Year-Of-Publication', 'N/A'),
                'publisher': r.get('Publisher', 'Unknown'),
                'description': desc
            })
        return results

    def _get_book_details(self, book_title):
        match = self.books_df[self.books_df['Book-Title'] == book_title]
        if not match.empty:
            row = match.iloc[0]
            desc = row.get('description') or row.get('Description') if ('description' in row or 'Description' in row) else None
            img = row.get('Image-URL-L') or row.get('Image-URL-M') or '/static/images/no-cover.jpg'
            return {
                'title': row['Book-Title'],
                'author': row['Book-Author'],
                'image': img,
                'year': row.get('Year-Of-Publication', 'N/A'),
                'publisher': row.get('Publisher', 'Unknown'),
                'description': desc
            }
        return {
            'title': book_title,
            'author': 'Unknown',
            'image': '/static/images/no-cover.jpg',
            'year': 'N/A',
            'publisher': 'Unknown',
            'description': None
        }


    def recommend(self, query, top_n=6):
        query = query.strip()
        if not query:
            return [], "Please enter a valid book title or keyword."

        # Case 1: Exact or Collaborative Match in pivot table
        # Find closest matching title in pt.index
        matching_titles = [t for t in self.pt.index if query.lower() in t.lower()]
        
        if matching_titles:
            target_title = matching_titles[0]
            idx = np.where(self.pt.index == target_title)[0][0]
            similar_items = sorted(list(enumerate(self.similarity_scores[idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]
            
            recommendations = []
            for item in similar_items:
                rec_title = self.pt.index[item[0]]
                details = self._get_book_details(rec_title)
                details['score'] = round(float(item[1]) * 100, 1)
                details['rec_type'] = 'Collaborative Recommendation'
                recommendations.append(details)

            return recommendations, f"Showing collaborative recommendations based on '{target_title}'"

        # Case 2: Content-Based TF-IDF Match across full dataset
        query_vec = self.tfidf_vectorizer.transform([query])
        sim_scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        top_indices = sim_scores.argsort()[::-1][:top_n]

        recommendations = []
        for idx in top_indices:
            score = sim_scores[idx]
            if score > 0:
                row = self.books_df.iloc[idx]
                img = row.get('Image-URL-L') or row.get('Image-URL-M') or '/static/images/no-cover.jpg'
                details = {
                    'title': row['Book-Title'],
                    'author': row['Book-Author'],
                    'image': img,
                    'year': row.get('Year-Of-Publication', 'N/A'),
                    'publisher': row.get('Publisher', 'Unknown'),
                    'score': round(float(score) * 100, 1),
                    'rec_type': 'Content & Keyword Match'
                }
                recommendations.append(details)

        if recommendations:
            return recommendations, f"Showing text search recommendations for '{query}'"
        else:
            return [], f"No matching books found for '{query}'. Try another search term!"

    def search_autocomplete(self, query, limit=8):
        query = query.strip().lower()
        if not query or len(query) < 2:
            return []
        
        matches = [t for t in self.all_titles if query in t.lower()][:limit]
        return matches

# Singleton instance
recommender_instance = None

def get_recommender():
    global recommender_instance
    if recommender_instance is None:
        recommender_instance = BookRecommender()
    return recommender_instance
