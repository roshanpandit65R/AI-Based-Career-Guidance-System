# 🎯 AI-Based Career Guidance System (Flask)

Welcome to the **AI-Powered Career Guidance System**, a web application built using **Flask** and **Machine Learning** to recommend suitable career paths based on a user's skills and interests — all without using any third-party APIs!

---


## 📌 Features

- ✅ Predicts careers based on user-entered **skills** and **interests**
- ✅ Trained on custom career dataset (editable)
- ✅ Simple and interactive frontend using HTML/CSS
- ✅ Completely **offline** ML model (no APIs used)
- ✅ Easily extensible to add more career paths and users

---

## 🛠️ Tech Stack

| Technology | Usage |
|------------|--------|
| Python | Backend & ML model |
| Flask | Web framework |
| HTML/CSS | Frontend |
| scikit-learn | Machine Learning |
| Pandas | Data handling |
| TfidfVectorizer + KNN | Recommendation logic |

---

## 🗂️ Folder Structure

```
career_guidance/
├── app.py                  # Flask backend
├── ml_model.py             # ML training script
├── career_data.csv         # Dataset for model training
├── career_model.pkl        # Trained model file
├── vectorizer.pkl          # Trained TF-IDF vectorizer
├── templates/
│   └── index.html          # Frontend HTML form
└── README.md               # Project documentation
```

---

## 📅 Installation & Setup

### 1. Clone the repository
```bash
git clone 
cd folder path 
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python ml_model.py
```

### 5. Run the Flask App
```bash
python app.py
```

### 6. Visit in browser
Open `http://127.0.0.1:5000/` in your browser.

---

## 🧠 How It Works

- User inputs **skills** and **interests**
- These are processed using **TF-IDF vectorization**
- A **K-Nearest Neighbors** model finds the most similar career path from the dataset
- Result is shown instantly without external API calls

---

## ✅ Future Enhancements

- Add NLP analysis for resume or paragraph-based input
- Store user profiles in a database
- Provide learning resources along with career suggestions
- Deploy on **Render, Railway, or HuggingFace Spaces**

---

## 📄 License

This project is licensed under the MIT License. You are free to use, modify, and distribute it.

---

