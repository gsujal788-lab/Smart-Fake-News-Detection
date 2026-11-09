# Smart Fake News Detection System using NLP

## Project Overview
This project detects whether a news article is likely to be Fake or Real using Natural Language Processing (NLP) and Machine Learning.

It uses TF-IDF for text feature extraction and compares two classification algorithms:
- Logistic Regression
- Multinomial Naive Bayes

## Student Details
- **Name:** Sujal Ghimire
- **Roll Number:** AIML-A6/SEP-11832
- **Program:** AI/ML with Python Training Program
- **Organization:** InternsElite

## Dataset
The project uses the ISOT Fake News Dataset:
- Fake news: `Fake.csv`
- Real news: `True.csv`

Dataset source:
- [Fake.csv](https://raw.githubusercontent.com/Reinforz/Fake-News-detection-with-ISOT-Dataset/main/Fake.csv)
- [True.csv](https://raw.githubusercontent.com/Reinforz/Fake-News-detection-with-ISOT-Dataset/main/True.csv)

Download both files and place them in the notebook's working directory before running the notebook.

The combined dataset contains 44,898 records:
- Fake: 23,481
- Real: 21,417

## Technologies Used
- Python
- Pandas and NumPy
- NLTK
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib

## Project Workflow
1. Load and combine the fake and real news datasets.
2. Assign labels: Fake = 0 and Real = 1.
3. Clean text using lowercase conversion, removal of non-alphabetic characters, stopword removal, and lemmatization.
4. Split the cleaned text into training and testing sets (80:20).
5. Fit TF-IDF on training text only and transform the test text.
6. Train Logistic Regression and Multinomial Naive Bayes.
7. Evaluate both models using accuracy, precision, recall, F1-score, and confusion matrices.
8. Save the trained models and TF-IDF vectorizer using Joblib.
9. Run the Streamlit application.

## Model Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | **98.71%** |
| Multinomial Naive Bayes | **93.44%** |

Logistic Regression performed better on the test dataset.

## Data Leakage Prevention
The dataset is split before TF-IDF fitting. The vectorizer learns its vocabulary and IDF values from the training text only. The test text is transformed using that already-fitted vectorizer. This prevents test-set information from leaking into training.

## Project Files
- `Smart_Fake_News_Detection.ipynb` — Main project notebook
- `app.py` — Streamlit application
- `logistic_regression_model.pkl` — Trained Logistic Regression model
- `naive_bayes_model.pkl` — Trained Naive Bayes model
- `tfidf_vectorizer.pkl` — Fitted TF-IDF vectorizer

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk streamlit joblib
```

### 2. Download the dataset
Download `Fake.csv` and `True.csv` from the links in the Dataset section and place them in the notebook's working directory.

### 3. Run the notebook
Open `Smart_Fake_News_Detection.ipynb` in Google Colab or Jupyter Notebook and run the cells in order.

### 4. Run the Streamlit application
Make sure `app.py` and all three `.pkl` files are in the same folder, then run:

```bash
streamlit run app.py
```

## Limitations
- Predictions depend on patterns learned from the training dataset.
- The system does not independently verify facts or news sources.
- Performance may vary on new topics, publishers, or writing styles.
- Predictions should not be treated as final proof that a news story is true or false.

## Future Improvements
- Use larger and more recent datasets.
- Add source credibility checks.
- Explore advanced NLP models.
- Improve explainability and external testing.

## Author
**Sujal Ghimire**

## Disclaimer
This project is developed for educational purposes as part of the InternsElite AI/ML with Python Training Program.
