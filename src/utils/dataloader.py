import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# https://towardsdatascience.com/feature-extraction-with-bert-for-text-classification-533dde44dc2f
def initialize_tokenizer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

    return device, tokenizer, model

def tokenize_df(df, tokenizer, model, device):
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

def get_bbc(wholeDataset = True):
    ## BBC dataset
    ## https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv
    df = pd.read_csv("../datasets/bbc-text.csv").sample(frac=1).head(100)

    ## preprocessing
    LE = LabelEncoder()
    df['label'] = LE.fit_transform(df['category'])

    df.head()
    
    print("Initializing Tokenizer...")

    device, tokenizer, model = initialize_tokenizer()

    if wholeDataset:
        return tokenize_df(df, tokenizer, model, device)
    else:
        ##dataset splitting... 80/20 rule
        df_train, df_test = train_test_split(df, int(len(df) * .8))

        df_train_x, df_train_y = tokenize_df(df_train, tokenizer, model, device)
        df_test_x, df_test_y = tokenize_df(df_test, tokenizer, model, device)

        return df_train_x, df_train_y, df_test_x, df_test_y
