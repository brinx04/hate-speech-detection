This project aims to detect hate speech in text data using Natural Language Processing (NLP) techniques. The end-to-end pipeline includes data exploration, text cleaning, feature extraction, model training, and evaluation. The key steps are:

Data Loading & EDA:
The dataset was loaded using Pandas, followed by initial exploration. The distribution of hate vs. non-hate speech classes was visualized using a bar chart to assess class imbalance.

Text Preprocessing:
Each text entry was tokenized using NLTK, cleaned by removing stopwords, and normalized using stemming (via PorterStemmer) or lemmatization (WordNetLemmatizer). These steps help reduce noise and standardize the text.

Feature Extraction:
Two approaches were used:

Bag of Words (BoW) with CountVectorizer

TF-IDF (Term Frequency-Inverse Document Frequency) with TfidfVectorizer, experimenting with unigram and bigram ranges
This transformed the raw text into numerical vectors suitable for machine learning.

Model Training & Evaluation:
Models such as Naive Bayes, Logistic Regression, or SVM were trained on both BoW and TF-IDF features. The results were evaluated using accuracy, precision, recall, and F1-score via classification_report. The impact of using BoW vs. TF-IDF was compared and discussed.

The final model can effectively classify hate vs. non-hate speech, demonstrating the importance of proper text preprocessing and feature engineering in NLP tasks.