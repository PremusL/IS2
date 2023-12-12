# %%
import pandas as pd
import numpy as np
import json
import ijson
import requests
from bs4 import BeautifulSoup
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# %%
#podatki iz jsona

file_path = "./News_Category_Dataset_IS_course.json"
data = [json.loads(line, object_hook=lambda o: str(o) if isinstance(o, (str, None))else o) for line in open(file_path, 'r')]

data

# %%
#naredi dataframe
df = pd.DataFrame(data)

categories = df['category'].value_counts()

num_categories = len(categories)
print(categories)

# %%
# adds the whole story to the dataframe

def add_story(df):
    text_column = []

    for i in range(15360, len(df)):
        # print(f"index: {i}")
        short_description = df['short_description'].iloc[i]

        if (type(short_description) != str):
            # print(short_description)
            print(f"index {i}")
            link = df['link'].iloc[i]
            try:
                response = requests.get(link)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # section = soup.find('section')
                    cur_text_arr = []
                    all_data_article = soup.find_all('div', class_='primary-cli') #all the text data in article that could be useful (the last few aren't)
                    for i in range(len(all_data_article) - 2):
                        k = all_data_article[i]
                        cur_text_arr.append(k.text)
                    current_string = " ".join(cur_text_arr)

                    df.at[i, 'short_description'] = current_string
            
            except:
                continue
                # text_column.append(current_string)

    

    # df['story'] = text_column
    return df




# links

stories = add_story(df)

stories


# %%
stories.to_csv("fixed_data.csv", index=False)

# %%
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()



# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text into words
    text = str(text)
    words = word_tokenize(text.lower())  # Convert text to lowercase

    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    words = [word.translate(table) for word in words if word.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Stemming (uncomment if you want to use stemming)
    stemmed_words = [stemmer.stem(word) for word in words]

    # Join the words back into a string
    preprocessed_text = ' '.join(lemmatized_words)
    return preprocessed_text

df = pd.read_csv('./fixed_data.csv', sep=',')
# df
df['cleaned_text'] = df['short_description'].apply(preprocess_text)

# %%
df.to_csv("fixed_data.csv", index=False)

# %%
data = pd.read_csv("./fixed_data.csv", sep=',')
data['cleaned_text'] = data['cleaned_text'].fillna('')
data['cleaned_text']



# %%

X_train, X_test, y_train, y_test = train_test_split(data[['cleaned_text', 'short_description', 'headline']], data['category'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()  # Use TF-IDF vectorizer for text to numerical feature conversion
X_train_vec = vectorizer.fit_transform(X_train['cleaned_text'])
X_test_vec = vectorizer.transform(X_test['cleaned_text'])

tokenized_train_text = [text.split() for text in X_train['cleaned_text']]
tokenized_test_text = [text.split() for text in X_test['cleaned_text']]


# tokenized_test_text

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Logistic Regression model
# logistic_model = LogisticRegression(max_iter=4000)
# logistic_model.fit(X_train_vec, y_train)
# logistic_predictions = logistic_model.predict(X_test_vec)
# logistic_accuracy = accuracy_score(y_test, logistic_predictions)
# print("Logistic Regression Accuracy:", logistic_accuracy)



# Random Forest model slabsi je
# rf_model = RandomForestClassifier()
# rf_model.fit(X_train_vec, y_train)
# rf_predictions = rf_model.predict(X_test_vec)
# rf_accuracy = accuracy_score(y_test, rf_predictions)
# print("Random Forest Accuracy:", rf_accuracy)

# %%
from gensim.models import Word2Vec
w2v_model = Word2Vec(tokenized_train_text, vector_size=100, window=5, min_count=1, workers=6, epochs=10)



# %%
all_words = w2v_model.wv.index_to_key


categories = [cat.lower() for cat in data['category'].unique().tolist()]
print(categories)

word_vectors_dict = {word: w2v_model.wv[word] for word in all_words}
# category_vectors = [w2v_model.wv[word] for word in categories]



# %%
word_vectors = [w2v_model.wv[word] for word in all_words]
print(word_vectors)


# logistic_model = LogisticRegression()
# logistic_model.fit(word_vectors, y_train)
# logistic_predictions = logistic_model.predict(X_test_vec)
# logistic_accuracy = accuracy_score(y_test, logistic_predictions)
# print("Logistic Regression Accuracy:", logistic_accuracy)

# %%
def text_to_vector(text, model):
    words = word_tokenize(text.lower())
    vectors = [model.wv[word] for word in words if word in model.wv]
    if not vectors:
        return None
    return sum(vectors) / len(vectors)

tokenized_texts = [word_tokenize(text.lower()) for text in data['cleaned_text']]
model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1, workers=6, epochs=10)

# Assuming 'texts' is a list of sentences
vectors = [text_to_vector(text, model) for text in data['cleaned_text']]


# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Assuming 'labels' is a list of class labels
cat = data['category']
# print(cat, len(cat), len(vectors))
filter_vec = list(filter(lambda v: v is not None, vectors))
filter_cat = [cat.iloc[i] for i,v in enumerate(vectors) if v is not None]
X_train, X_test, y_train, y_test = train_test_split(filter_vec, np.array(filter_cat), test_size=0.2, random_state=42)

# Train a classifier
classifier = LogisticRegression(max_iter=4000)
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)



