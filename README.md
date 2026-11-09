# Smart Fake News Detection System using NLP

## Project Overview
The Smart Fake News Detection System is a machine learning project that classifies news articles as Fake or Real using Natural Language Processing (NLP).

The project uses TF-IDF for text feature extraction and compares two machine learning algorithms: Logistic Regression and Multinomial Naive Bayes.

## Objectives
- Detect potentially fake and real news articles.
- Apply NLP techniques to clean and process text.
- Convert text into numerical features using TF-IDF.
- Train and compare machine learning models.
- Build an interactive web application using Streamlit.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Google Colab

## Dataset
The project uses the ISOT Fake News Dataset.

- Fake.csv – Fake news articles.
- True.csv – Real news articles.

**Dataset statistics:**
- Total articles: 44,898
- Fake articles: 23,481
- Real articles: 21,417
- Missing values: 0

## Methodology

### 1. Data Preprocessing
- Converted text to lowercase.
- Removed special characters and numbers.
- Removed English stopwords.
- Applied lemmatization.
- Combined news titles and article text.

### 2. Feature Extraction
TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert text into numerical features.

Maximum features: 5,000.

### 3. Train-Test Split
- Training data: 80%
- Testing data: 20%

### 4. Machine Learning Models
Two algorithms were trained:
- Logistic Regression
- Multinomial Naive Bayes

## Model Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | 98.70% |
| Multinomial Naive Bayes | 93.46% |

Logistic Regression achieved the higher accuracy on the test dataset and was selected as the primary model for the Streamlit application.

## Streamlit Web Application
The application allows users to enter a news article and receive:
- Predicted category: Fake News or Real News.
- Model confidence percentage.

The prediction is based on patterns learned from the training dataset and should not be treated as independent factual verification.

## Project Files
- `Smart_Fake_News_Detection.ipynb` – Complete Python notebook.
- `app.py` – Streamlit web application.
- `logistic_regression_model.pkl` – Trained Logistic Regression model.
- `naive_bayes_model.pkl` – Trained Naive Bayes model.
- `tfidf_vectorizer.pkl` – Saved TF-IDF vectorizer.

## How to Run the Project

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn streamlit joblib
```

### 2. Run the Streamlit Application

Make sure all three model files and `app.py` are in the same folder.

```bash
streamlit run app.py
```

### 3. Use the Application
1. Open the Streamlit application in your browser.
2. Enter a news article.
3. Click Detect News.
4. View the predicted category and confidence percentage.

## Limitations
- Predictions depend on the dataset used for training.
- The model may not perform equally well on new or unrelated news sources.
- High confidence does not guarantee that a news article is factually true or false.
- The system should be used as a classification aid, not a replacement for fact-checking.

## Future Improvements
- Train using more diverse and recent news datasets.
- Explore advanced NLP and deep learning models.
- Improve performance on unseen news sources.
- Add source verification and fact-checking features.

## Author
Sujal Ghimire

## Internship Program
AI/ML with Python Training Program – InternsElite
