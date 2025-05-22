#train_data is labeled
#test is unlabeled
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

#lemmatizer-reduces word to base form
#stemming- chops of prefixes and suffixes

nltk.download('punkt')#word tokenizer
nltk.download('wordnet')#required for lemmatizer(for base form)
nltk.download('omw-1.4')#for lemmatizer
nltk.download('stopwords')#to remove common words
nltk.download('punkt_tab')#used by punkt
train_df=pd.read_csv('train_data.csv')
test_df=pd.read_csv('test.csv')

print(train_df.head())
print(test_df.head())
print(train_df['HS'].value_counts())
train_df['HS'].value_counts().plot(kind='bar',title='Class Distribution')
plt.xlabel('HS Label')
plt.ylabel('Count')
plt.show()

def tokenize(text):
  return word_tokenize(text.lower())
test_df['tokenized_text']=test_df['text'].apply(tokenize)

#logisticregression does classification and predicts discrete labels
#naive bayes works on bayes theorem

stop_words=set(stopwords.words('english'))

def remove_stopwords(tokens):
  return [word for word in tokens if word not in stop_words]

lemmatizer=WordNetLemmatizer()

def lemmatize(tokens):
  return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess(text):
  tokens=tokenize(text)
  tokens=remove_stopwords(tokens)
  tokens=lemmatize(tokens)
  return ' '.join(tokens)

#BoW is a way to convert text into numbers

train_df['clean_text']=train_df['text'].apply(preprocess)
test_df['clean_text']=test_df['text'].apply(preprocess)

bow_vectorizer=CountVectorizer(max_features=5000)#word becomes vector of counts(no. of times a word appear)
X_train_bow=bow_vectorizer.fit_transform(train_df['clean_text'])#converts training text to vectors
X_test_bow=bow_vectorizer.transform(test_df['clean_text'])#same process

#tfidf(term frequency inverse document frequency) converts text into numerical features
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)#uni and bigrams conv text into weighted text
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['clean_text'])#transform train set
X_test_tfidf = tfidf_vectorizer.transform(test_df['clean_text'])

y_train=train_df['HS']#target label

low_reg_bow=LogisticRegression().fit(X_train_bow,y_train)#trains logicregression on BoW and TFIDF
low_reg_tfidf=LogisticRegression().fit(X_train_tfidf,y_train)

nb_bow=MultinomialNB().fit(X_train_bow,y_train)#trains naive bayes on BoW and TFIDF
nb_tfidf=MultinomialNB().fit(X_train_tfidf,y_train)

X_train,x_val,y_train,y_val=train_test_split(X_train_tfidf,y_train,test_size=0.2)#splits data into 80 training and 20 validation
model=MultinomialNB().fit(X_train,y_train)#new training split
y_pred=model.predict(x_val)#predicts validation data
print(classification_report(y_val,y_pred))#prints precision recall and F1-score

final_preds=nb_tfidf.predict(X_test_tfidf)
submission_df=pd.DataFrame({'id':test_df['id'],'HS':final_preds})
submission_df.to_csv('submission.csv',index=False)