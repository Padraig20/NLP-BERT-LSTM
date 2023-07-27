import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import nltk
from nltk import ngrams
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import random
import numpy as np
import torchtext
from torchtext.data.utils import get_tokenizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# https://towardsdatascience.com/feature-extraction-with-bert-for-text-classification-533dde44dc2f
def initialize_bert_tokenizer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    return device, tokenizer, model

def tokenize_df_bert_hiddenstates(df, tokenizer, model, device):
    tokenized = tokenizer(df["text"].values.tolist(), padding=True, truncation=True, return_tensors="pt")

    print(tokenized.keys())

    # move on device (GPU)
    # tokenized = {k:torch.tensor(v).to(device) for k,v in tokenized.items()}

    with torch.no_grad():
        hidden = model(**tokenized)  # dim : [batch_size(nr_sentences), tokens, emb_dim]
        embeddings = hidden.last_hidden_state #dim : [batch_size, sequence_length, hidden_size]

    y = df["label"]

    return embeddings, y

def tokenize_df_bert(df, tokenizer, model, device):
    tokenized = [tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for text in df['text']]
    x = tokenized
    y = df['label'].tolist()

    print(len(x), len(y))

    return x, y

def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test

def generate_ngrams(text, n):
    # tokenize input text
    tokens = nltk.word_tokenize(text)

    n_grams = list(ngrams(tokens, n))
    return n_grams
    
def synonym_replacement(text, n=1):
    words = nltk.word_tokenize(text)
    augmented_texts = []
    for _ in range(n):
        augmented_words = []
        for word in words:
            synsets = wordnet.synsets(word)
            if synsets:
                synonym = random.choice(synsets).lemmas()[0].name()
                augmented_words.append(synonym)
            else:
                augmented_words.append(word)
        augmented_texts.append(" ".join(augmented_words))
    return augmented_texts

def augment_data(data, target_class_count):
    augmented_data = pd.DataFrame(columns=data.columns)

    for category in data['category'].unique():
        category_data = data[data['category'] == category]
        category_count = len(category_data)

        if category_count >= target_class_count:
            augmented_data = pd.concat([augmented_data, category_data.sample(target_class_count, replace=False)], ignore_index=True)
        else:
            augment_count = target_class_count - category_count

            augmented_texts = []
            for i in range(augment_count):
                augmented_texts.extend(synonym_replacement(category_data.iloc[i % category_count]['text']))

            augmented_df = pd.DataFrame(augmented_texts, columns=['text'])
            augmented_df['category'] = category

            augmented_data = pd.concat([augmented_data, category_data, augmented_df], ignore_index=True)

    return augmented_data

def get_bbc_dataset_augmented(data):
    # Class count to balance the dataset (you can adjust this value as needed)
    target_class_count = data['category'].value_counts().max()

    print(data['category'].value_counts())

    # Augment the data to remove class imbalances
    augmented_data = augment_data(data, target_class_count)

    # Print the class distribution after augmentation (optional)
    print(augmented_data['category'].value_counts())

    return augmented_data

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

def get_bbc_tokenized_bert(wholeDataset = True, hiddenState = False, augmented = False):
    ## BBC dataset
    ## https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv
    df = pd.read_csv("../datasets/bbc-text.csv").sample(frac=1).head(50)

    if augmented:
        df = get_bbc_dataset_augmented(df)

    ## preprocessing
    df['text'] = remove_stop_words(df['text'])

    LE = LabelEncoder()
    df['label'] = LE.fit_transform(df['category'])

    encoded_labels = df['label']
    original_labels = LE.inverse_transform(encoded_labels)
    label_encoding_map = dict(zip(original_labels, encoded_labels))

    print()
    print(label_encoding_map)
    print()
    print(df.head())
    
    print("Initializing Tokenizer...")

    device, tokenizer, model = initialize_bert_tokenizer()

    if wholeDataset:
        if hiddenState:
            return tokenize_df_bert_hiddenstates(df, tokenizer, model, device)
        else:
            return tokenize_df_bert(df, tokenizer, model, device)
    else:
        ##dataset splitting... 80/20 rule
        df_train, df_test = train_test_split(df, int(len(df) * .8))

        if hiddenState:
            df_train_x, df_train_y = tokenize_df_bert_hiddenstates(df_train, tokenizer, model, device)
            df_test_x, df_test_y = tokenize_df_bert_hiddenstates(df_test, tokenizer, model, device)

        else:
            df_train_x, df_train_y = tokenize_df_bert(df_train, tokenizer, model, device)
            df_test_x, df_test_y = tokenize_df_bert(df_test, tokenizer, model, device)

        return df_train_x, df_train_y, df_test_x, df_test_y

def get_bbc_tokenized_ngrams(wholeDataset = True, n = 2, augmented = False): # standard: bigrams
    ## BBC dataset
    ## https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv
    
    df = pd.read_csv("../datasets/bbc-text.csv").head(50)

    if augmented:
        df = get_bbc_dataset_augmented(df)

    ## preprocessing
    df['text'] = remove_stop_words(df['text'])

    LE = LabelEncoder()
    df['label'] = LE.fit_transform(df['category'])

    encoded_labels = df['label']
    original_labels = LE.inverse_transform(encoded_labels)
    label_encoding_map = dict(zip(original_labels, encoded_labels))

    print()
    print(label_encoding_map)
    print()
    print(df.head())
    
    print("Initializing Tokenizer...")

    n_grams_list = [generate_ngrams(text, n) for text in df['text']]

    # convert to text for CountVectorizer
    n_grams_text = [' '.join([' '.join(gram) for gram in grams]) for grams in n_grams_list]

    ## initialize vectorizer
    vectorizer = CountVectorizer()
    feature_matrix = vectorizer.fit_transform(n_grams_text)

    if (wholeDataset):
        return feature_matrix, df['label']
    else:
        df_train_x, df_test_x = train_test_split(feature_matrix, int(feature_matrix.shape[0] * .8))
        df_train_y, df_test_y = train_test_split(df['label'], int(len(df['label']) * .8))

        return df_train_x, df_train_y, df_test_x, df_test_y

def get_bbc_tokenized_torch(wholeDataset = True, augmented=False):
    ## BBC dataset
    ## https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv
    df = pd.read_csv("../datasets/bbc-text.csv").sample(frac=1).head(50)

    if augmented:
        df = get_bbc_dataset_augmented(df)

    ## preprocessing
    df['text'] = remove_stop_words(df['text'])

    LE = LabelEncoder()
    df['label'] = LE.fit_transform(df['category'])

    encoded_labels = df['label']
    original_labels = LE.inverse_transform(encoded_labels)
    label_encoding_map = dict(zip(original_labels, encoded_labels))

    print()
    print(label_encoding_map)
    print()
    print(df.head())

    tokenizer = get_tokenizer("basic_english")

    df['tokens'] = df['text'].apply(tokenizer)
    
    if (wholeDataset):
        return df['tokens'], df['label']
    else:
        df_train_x, df_test_x = train_test_split(df['tokens'], int(len(df['tokens']) * .8))
        df_train_y, df_test_y = train_test_split(df['label'], int(len(df['label']) * .8))

        return df_train_x, df_train_y, df_test_x, df_test_y

def get_bbc_vanilla(wholeDataset = True, augmented=False):
    ## BBC dataset
    ## https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv
    df = pd.read_csv("../datasets/bbc-text.csv").sample(frac=1).head(50)

    if augmented:
        df = get_bbc_dataset_augmented(df)

    ## preprocessing
    df['text'] = remove_stop_words(df['text'])

    LE = LabelEncoder()
    df['label'] = LE.fit_transform(df['category'])

    encoded_labels = df['label']
    original_labels = LE.inverse_transform(encoded_labels)
    label_encoding_map = dict(zip(original_labels, encoded_labels))

    print()
    print(label_encoding_map)
    print()
    print(df.head())
    
    if (wholeDataset):
        return df['text'], df['label']
    else:
        df_train_x, df_test_x = train_test_split(df['text'], int(len(df['text']) * .8))
        df_train_y, df_test_y = train_test_split(df['label'], int(len(df['label']) * .8))

        return df_train_x, df_train_y, df_test_x, df_test_y
