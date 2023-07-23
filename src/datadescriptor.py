import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import numpy as np

def plot_bbc_dataset_characteristics():
    dataset_path = '../../datasets/bbc-text.csv'
    data = pd.read_csv(dataset_path)

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

    # Class Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='category', data=data, palette='viridis')
    plt.xticks(rotation=45)
    plt.title('Class Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Average Text Length per Category
    plt.figure(figsize=(10, 6))
    sns.barplot(x='category', y='text_length', data=data, palette='viridis')
    plt.xticks(rotation=45)
    plt.title('Average Text Length per Category')
    plt.xlabel('Category')
    plt.ylabel('Average Text Length')
    plt.tight_layout()
    plt.show()

import sys

if len(sys.argv) != 2:
    print("Usage:\t\tpython datadescriptor.py (\"bbc\"|\"\")")
    sys.exit(1)

mode = sys.argv[1]

if mode == "bbc":
    plot_bbc_dataset_characteristics()
else:
    print("Unknown mode!")
    print("Usage:\t\tpython datadescriptor.py (\"bbc\"|\"\")")
    sys.exit(1)
