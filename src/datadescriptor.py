import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def remove_stop_words(data):
    cleaned_data = []
    for sentences in data:       
        tokenized = word_tokenize(sentences, 'english')
        new_sentence =  [token for token in tokenized if token not in stopwords.words('english') and token != 'said' and token != 'will']
        sent = ''
        for text in new_sentence:
            sent+=text + ' '
        cleaned_data.append(sent)
    return np.array(cleaned_data)


def plot_bbc_dataset_characteristics(stripped = False):
    dataset_path = '../datasets/bbc-text.csv'
    data = pd.read_csv(dataset_path)
    
    if stripped:
        data['text'] = remove_stop_words(data['text'])

    # plot the number of samples per category
    plt.figure(figsize=(10, 6))
    sns.countplot(x='category', data=data, palette='viridis')
    plt.xticks(rotation=45)
    plt.title('Number of Samples per Category')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # plot the distribution of text lengths
    data['text_length'] = data['text'].apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(data['text_length'], kde=True, color='purple')
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Text Length')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Word Frequency
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['text'])
    word_freq = np.asarray(X.sum(axis=0)).squeeze()
    word_freq_df = pd.DataFrame({'word': vectorizer.get_feature_names_out(), 'frequency': word_freq})
    word_freq_df = word_freq_df.sort_values(by='frequency', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='word', y='frequency', data=word_freq_df.head(20), palette='viridis')
    plt.xticks(rotation=45)
    plt.title('Top 20 Most Frequent Words')
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Word Clouds for each category
    plt.figure(figsize=(12, 8))
    unique_categories = data['category'].unique()
    for i, category in enumerate(unique_categories):
        plt.subplot(2, 3, i + 1)
        category_text = " ".join(data[data['category'] == category]['text'])
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(category_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(category)
    plt.tight_layout()
    plt.show()

import sys

if len(sys.argv) != 2:
    print("Usage:\t\tpython datadescriptor.py (\"bbc\"|\"bbc_stripped\")")
    sys.exit(1)

mode = sys.argv[1]

if mode == "bbc":
    plot_bbc_dataset_characteristics()
elif mode == "bbc_stripped":
    plot_bbc_dataset_characteristics(True)
else:
    print("Unknown mode!")
    print("Usage:\t\tpython datadescriptor.py (\"bbc\"|\"bbc_stripped\"|\"\")")
    sys.exit(1)
