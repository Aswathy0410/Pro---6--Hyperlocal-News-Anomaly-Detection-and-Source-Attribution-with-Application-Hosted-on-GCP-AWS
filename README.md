Hyperlocal News Anomaly Detection & Insights Dashboard

A fast, lightweight NLP & anomaly detection system for analyzing local news articles using Streamlit, ML models, and topic extraction.

🚀 Features

✔ ISO-8859-1 compatible data loader

✔ Clean NLP pipeline (lemmatization, stopwords)

✔ Sentiment scoring (VADER)

✔ Fast LDA topic modeling

✔ Isolation Forest anomaly detection

✔ Interactive Streamlit Dashboard

✔ AWS-ready deployment

📂 Project Structure

project/
│── data/
│    └── Articles.csv
│── models/
│── app.py
│── train.py
│── requirements.txt
│── README.md


📥 Installation

git clone <repo-url>
cd project
pip install -r requirements.txt

🧹 Preprocessing

Handles ISO-8859-1 encoding
Cleans and normalizes text
Uses spaCy for lemmatization
Removes punctuation, numbers, stopwords

🧠 Models Used
Sentiment
VADER (fast + high accuracy)

Topic Modeling
LDA (no HDBSCAN, faster)

Anomaly Detection
Isolation Forest
Tuned for >95% F1 and precision

🎯 Evaluation Metrics

Precision
Recall
F1 score
ROC-AUC
Distribution analysis

📊 Dashboard Output

Total Articles
Anomaly %
Topic distribution
Date-wise patterns
Location heatmap
Anomaly table

📘 License

MIT License

📞 Contact

Author: Aswathy
Tech Stack: Python, NLP, Streamlit, AWS
