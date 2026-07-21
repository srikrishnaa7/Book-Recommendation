# 📚 Book Recommendation System

A content-based Book Recommendation System built using **Python**, **Flask**, **Pandas**, and **Scikit-learn**. The application recommends books similar to the one selected by the user using machine learning techniques.

---

## 🚀 Features

- 📖 Recommend books based on similarity
- 🔍 Search for books by title
- 🖼️ Display book covers
- ⭐ Show top recommended books
- 🌐 Simple and responsive Flask web interface
- ⚡ Fast recommendation using precomputed similarity matrix

---

## 🛠️ Tech Stack

### Backend
- Python
- Flask

### Machine Learning
- Pandas
- NumPy
- Scikit-learn

### Frontend
- HTML
- CSS
- Bootstrap

### Dataset
- Book-Crossing Dataset

---

## 📂 Project Structure

```
Book-Recommendation/
│
├── static/
│   ├── css/
│   └── images/
│
├── templates/
│   ├── index.html
│   └── recommend.html
│
├── app.py
├── requirements.txt
├── popular.pkl
├── pt.pkl
├── books.pkl
├── similarity_scores.pkl
└── README.md
```

---

## ⚙️ How It Works

1. Load the processed book dataset.
2. Create a user-book matrix.
3. Calculate similarity between books using Cosine Similarity.
4. When a user selects a book, retrieve the most similar books.
5. Display the recommended books along with their cover images.

---

## 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/srikrishnaa7/Book-Recommendation.git
```

### 2. Navigate to the project

```bash
cd Book-Recommendation
```

### 3. Create a virtual environment (Optional)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---


## 📊 Machine Learning Approach

The recommendation engine uses **Collaborative Filtering** based on user ratings.

Steps involved:

- Data Cleaning
- Filtering active users
- Creating User-Item Matrix
- Pivot Table Generation
- Cosine Similarity
- Recommendation Generation

---

## 📦 Requirements

```
Flask
NumPy
Pandas
Scikit-learn
Pickle
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🎯 Future Improvements

- User Login & Authentication
- Personalized recommendations
- Book genre filtering
- Search autocomplete
- Recommendation history
- Deploy on Render / Railway / AWS

---

## 👨‍💻 Author

**Sri Krishna**

- GitHub: https://github.com/srikrishnaa7

---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub and feel free to contribute!

---

## 📜 License

This project is for educational purposes.
