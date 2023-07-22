import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# https://towardsdatascience.com/feature-extraction-with-bert-for-text-classification-533dde44dc2f
def initialize_bert_tokenizer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

    return device, tokenizer, model

def tokenize_df_bert(df, tokenizer, model, device):
    tokenized = tokenizer(df["text"].values.tolist(), padding=True, truncation=True, return_tensors="pt")

    print(tokenized.keys())

    # move on device (GPU)
    # tokenized = {k:torch.tensor(v).to(device) for k,v in tokenized.items()}

    with torch.no_grad():
        hidden = model(**tokenized)  # dim : [batch_size(nr_sentences), tokens, emb_dim]

    # get only the [CLS] hidden states
    cls = hidden.last_hidden_state[:, 0, :]

    x = cls.to("cpu")
    y = df["label"]

    print(x.shape, y.shape)

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

def get_bbc_tokenized_bert(wholeDataset = True):
    ## BBC dataset
    ## https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv
    df = pd.read_csv("../datasets/bbc-text.csv").sample(frac=1).head(100)

    ## preprocessing
    LE = LabelEncoder()
    df['label'] = LE.fit_transform(df['category'])

    df.head()
    
    print("Initializing Tokenizer...")

    device, tokenizer, model = initialize_bert_tokenizer()

    if wholeDataset:
        return tokenize_df_bert(df, tokenizer, model, device)
    else:
        ##dataset splitting... 80/20 rule
        df_train, df_test = train_test_split(df, int(len(df) * .8))

        df_train_x, df_train_y = tokenize_df_bert(df_train, tokenizer, model, device)
        df_test_x, df_test_y = tokenize_df_bert(df_test, tokenizer, model, device)

        return df_train_x, df_train_y, df_test_x, df_test_y

def get_bbc_tokenized_ngrams(wholeDataset = True, n = 2): # standard: bigrams
    ## BBC dataset
    ## https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv
    df = pd.read_csv("../datasets/bbc-text.csv").sample(frac=1).head(100)

    # Download the 'punkt' resource for tokenization
    nltk.download('punkt')

    # Download the 'stopwords' resource for filtering stop words
    nltk.download('stopwords')

    ## preprocessing
    LE = LabelEncoder()
    df['label'] = LE.fit_transform(df['category'])

    df.head()
    
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




