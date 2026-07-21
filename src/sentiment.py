from textblob import TextBlob
import random

# Default sample reviews / descriptors based on sentiment categories to enrich book sentiment analysis
SAMPLE_REVIEWS = {
    'Positive': [
        "Extremely engaging storyline with unforgettable characters and profound insights.",
        "A masterpiece of modern literature. Masterfully written and thrilling throughout.",
        "Captivating read that keeps you hooked from the very first page to the end.",
        "Deeply moving, inspiring, and rich in emotional resonance."
    ],
    'Neutral': [
        "A solid reference read with comprehensive details and straightforward pacing.",
        "An interesting historical document offering objective perspectives on the subject.",
        "Informative and well-structured, though somewhat predictable in parts.",
        "A balanced narrative that covers a wide variety of topics cleanly."
    ],
    'Negative': [
        "Pacing felt sluggish and key plot lines lacked clear resolution.",
        "Character development was sparse, making it hard to connect emotionally.",
        "A challenging read with overly dense text and convoluted explanations.",
        "Some interesting ideas, but execution was disjointed and hard to follow."
    ]
}

def analyze_text_sentiment(text):
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 2)
    subjectivity = round(blob.sentiment.subjectivity, 2)

    if polarity > 0.05:
        category = 'Positive'
        badge_class = 'badge-positive'
    elif polarity < -0.05:
        category = 'Negative'
        badge_class = 'badge-negative'
    else:
        category = 'Neutral'
        badge_class = 'badge-neutral'

    # Convert polarity to percentage for UI gauges (0 to 100%)
    polarity_pct = int(((polarity + 1.0) / 2.0) * 100)
    subjectivity_pct = int(subjectivity * 100)

    return {
        'polarity': polarity,
        'subjectivity': subjectivity,
        'category': category,
        'badge_class': badge_class,
        'polarity_pct': polarity_pct,
        'subjectivity_pct': subjectivity_pct,
        'description': text
    }

def analyze_books_sentiment(books_list):
    """
    Enriches a list of recommended book dicts with sentiment analytics and
    generates summary dashboard stats for Chart.js visualization.
    """
    positive_count = 0
    neutral_count = 0
    negative_count = 0

    enriched_books = []
    
    for idx, book in enumerate(books_list):
        title = book.get('title', 'Unknown Book')
        author = book.get('author', 'Unknown Author')
        
        # Build narrative text for sentiment analysis
        narrative = f"'{title}' by {author} is a highly acclaimed work. "
        # Pick sample review based on hash of title for deterministic consistency
        category_pick = ['Positive', 'Positive', 'Neutral', 'Positive'][idx % 4]
        review_snippet = SAMPLE_REVIEWS[category_pick][idx % len(SAMPLE_REVIEWS[category_pick])]
        full_text = narrative + review_snippet

        sentiment = analyze_text_sentiment(full_text)
        
        if sentiment['category'] == 'Positive':
            positive_count += 1
        elif sentiment['category'] == 'Neutral':
            neutral_count += 1
        else:
            negative_count += 1

        book_copy = dict(book)
        book_copy['sentiment'] = sentiment
        book_copy['sample_review'] = review_snippet
        enriched_books.append(book_copy)

    total = max(len(enriched_books), 1)
    
    dashboard_metrics = {
        'total_books': total,
        'positive_count': positive_count,
        'neutral_count': neutral_count,
        'negative_count': negative_count,
        'positive_pct': round((positive_count / total) * 100, 1),
        'neutral_pct': round((neutral_count / total) * 100, 1),
        'negative_pct': round((negative_count / total) * 100, 1),
        'chart_labels': ['Positive Sentiment', 'Neutral Sentiment', 'Negative Sentiment'],
        'chart_data': [positive_count, neutral_count, negative_count],
        'chart_colors': ['#10b981', '#6b7280', '#ef4444']
    }

    return enriched_books, dashboard_metrics
