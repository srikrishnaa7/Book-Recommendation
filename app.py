import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from src.recommender import get_recommender
from src.sentiment import analyze_books_sentiment

app = Flask(__name__)

# Pre-initialize recommender instance at startup
recommender = get_recommender()

@app.route('/')
def index():
    popular_books = recommender.get_popular_books(top_n=50)
    return render_template('index.html', popular_books=popular_books)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        limit = int(request.form.get('limit', 10))
    else:
        query = request.args.get('query', '').strip()
        limit = int(request.args.get('limit', 10))

    if not query:
        return render_template('recommendations.html', query='', books=[], metrics=None, message="Please enter a book title or keyword to search.", limit=limit)

    raw_recommendations, message = recommender.recommend(query=query, top_n=limit)
    
    if not raw_recommendations:
        return render_template('recommendations.html', query=query, books=[], metrics=None, message=message, limit=limit)

    enriched_books, dashboard_metrics = analyze_books_sentiment(raw_recommendations)

    return render_template(
        'recommendations.html',
        query=query,
        books=enriched_books,
        metrics=dashboard_metrics,
        message=message,
        limit=limit
    )

@app.route('/sentiment_analysis', methods=['GET'])
def sentiment_analysis():
    query = request.args.get('query', '').strip()
    default_limit = 12 if query else 50
    limit = int(request.args.get('limit', default_limit))
    
    if not query:
        # Load popular books up to the specified limit (default 50)
        popular_books = recommender.get_popular_books(top_n=limit)
        enriched_books, dashboard_metrics = analyze_books_sentiment(popular_books)
        message = f"Displaying sentiment analysis for top {len(enriched_books)} popular books."
    else:
        raw_recommendations, message = recommender.recommend(query=query, top_n=limit)
        if not raw_recommendations:
            raw_recommendations = recommender.get_popular_books(top_n=limit)
        enriched_books, dashboard_metrics = analyze_books_sentiment(raw_recommendations)

    return render_template(
        'sentiment_analysis.html',
        query=query,
        books=enriched_books,
        metrics=dashboard_metrics,
        message=message,
        limit=limit
    )


@app.route('/api/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('q', '').strip()
    if len(query) < 2:
        return jsonify({'titles': []})
    
    matches = recommender.search_autocomplete(query=query, limit=8)
    return jsonify({'titles': matches})

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("[Flask] Starting Book Recommendation server...")
    app.run(debug=True, port=5000)
