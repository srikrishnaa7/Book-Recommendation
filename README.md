# 📚 Book Recommendation System

A Machine Learning based Book Recommendation System developed using **Python** and **Flask**. The application recommends books similar to the selected book using collaborative filtering and cosine similarity.

---

## 🚀 Features

- 📖 Recommend books based on user ratings
- 🔍 Search books by title
- ⭐ Display similar books instantly
- 🖼️ Show book cover images
- 🌐 Simple and interactive web interface

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

### Deployment
- Render

---

## 📂 Project Structure

```
Book-Recommendation/
│
├── data/                  # Dataset files
├── src/                   # Model training and preprocessing scripts
├── static/                # CSS, images and static files
├── templates/             # HTML templates
│
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── render.yaml            # Render deployment configuration
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/srikrishnaa7/Book-Recommendation.git
```

```bash
cd Book-Recommendation
```

---

### Create a Virtual Environment (Optional)

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Run the Application

```bash
python app.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

---

## 🧠 Recommendation Algorithm

This project uses **Collaborative Filtering** to recommend books.

The workflow includes:

- Data Cleaning
- User Rating Filtering
- User-Book Pivot Table
- Cosine Similarity Calculation
- Top Similar Book Recommendations

---

## 📁 Dataset

The project uses the **Book-Crossing Dataset**, which contains:

- Books
- Users
- Ratings

---


## 📌 Future Improvements

- User authentication
- Personalized recommendations
- Genre-based filtering
- Search autocomplete
- Book details page
- Dark mode
- Deploy with Docker

---

## 👨‍💻 Author

**Sri Krishna**

- GitHub: https://github.com/srikrishnaa7

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push to the branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## ⭐ Show your support

If you found this project useful, consider giving it a ⭐ on GitHub.

---

## 📄 License

This project is licensed under the MIT License.
