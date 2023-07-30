from collections import Counter
import torchtext
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
import numpy as np
from utils.dataloader import get_bbc_vanilla
from utils.dataloader import get_bbc_tokenized_bert
from utils.dataloader import get_bbc_tokenized_torch
from utils.dataloader import get_spam_vanilla
from utils.dataloader import get_spam_tokenized_bert
from utils.dataloader import get_spam_tokenized_torch
import torch
from torch import nn
from transformers import BertModel
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import sys

def plot_statistics(train_acc, train_loss, test_acc, test_loss):
    epochs = list(range(1, len(train_acc) + 1))

    plt.plot(epochs, train_acc, marker='o', label='train')
    plt.plot(epochs, test_acc, marker='o', label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.grid(True)
    plt.xticks(epochs)
    plt.legend()
    plt.show()

    plt.plot(epochs, train_loss, marker='o', label='train')
    plt.plot(epochs, test_loss, marker='o', label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.grid(True)
    plt.xticks(epochs)
    plt.legend()
    plt.show()

class Dataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_seq_length=None):
        self.texts = texts
        self.labels = labels

        # If vocab is not provided, build it from the texts
        if vocab is None:
            self.vocab = build_vocab_from_iterator(self.texts, specials=["<pad>", "<unk>"])
        else:
            self.vocab = vocab

        self.word_to_idx = self.vocab.get_stoi()
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # convert words to indices via vocabulary
        text_indices = [self.word_to_idx[word] if word in self.word_to_idx else self.word_to_idx["<unk>"] for word in text]

        # pad sequences to the same length
        if self.max_seq_length is not None:
            if len(text_indices) < self.max_seq_length:
                text_indices = text_indices + [self.word_to_idx["<pad>"]] * (self.max_seq_length - len(text_indices))
            else:
                text_indices = text_indices[:self.max_seq_length]

        return torch.tensor(text_indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def collate_fn(self, batch):
        texts, labels = zip(*batch)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=self.word_to_idx["<pad>"])
        return padded_texts, torch.stack(labels)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_output, _ = self.lstm(embedded)
        lstm_output = lstm_output[:, -1, :]  # Take the last output of the LSTM sequence
        return self.fc(lstm_output)

import argparse

parser = argparse.ArgumentParser(description='LSTM based deeplearning.')

parser.add_argument('-a', '--augmentation', type=bool, default=False,
                    help='Choose whether data augmentation should be performed before training.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                    help='Choose learning rate of the LSTM model.')
parser.add_argument('-s', '--embedding_dim', type=int, default=150,
                    help='Choose embedding size of the LSTM model.')
parser.add_argument('-l', '--layers', type=int, default=256,
                    help='Choose number of layers of the LSTM model.')
parser.add_argument('-e', '--epochs', type=int, default=5,
                    help='Choose epochs of the LSTM model.')
parser.add_argument('-m', '--max_seq_length', type=int, default=512,
                    help='Choose maximum sequence length for the vocabulary.')
parser.add_argument('-d', '--hidden_dim', type=int, default=5,
                    help='Choose hidden size of the LSTM model.')
parser.add_argument('dataset', type=str, help='"bbc" | "spam"')

args = parser.parse_args()

lr = args.learning_rate
augmentation = args.augmentation
hidden_dim = args.hidden_dim
num_layers = args.layers
epochs = args.epochs
dataset = args.dataset
max_seq_length = args.max_seq_length
embedding_dim = args.embedding_dim

if dataset == "bbc":
    df_train_x, df_train_y, df_test_x, df_test_y = get_bbc_tokenized_torch(False, augmentation)
    output_size = 5 # the number of output classes
elif dataset == "spam":
    df_train_x, df_train_y, df_test_x, df_test_y = get_spam_tokenized_torch(False, augmentation)
    output_size = 2 # the number of output classes
else:
    print("\t\tUnknown dataset, choose either bbc or spam.")
    sys.exit(1)

train = Dataset(df_train_x.values, df_train_y.values, max_seq_length=max_seq_length)
test = Dataset(df_test_x.values, df_test_y.values, max_seq_length=max_seq_length)

input_size = len(train.vocab)

model = LSTMModel(input_size, embedding_dim, hidden_dim, output_size, num_layers)

train_dataloader = DataLoader(train, batch_size=64, shuffle=True) #add randomness to training data
val_dataloader = DataLoader(test, batch_size=64)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)

train_loss = []
train_acc = []
test_loss = []
test_acc = []

if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()
for epoch_num in range(epochs):
    total_acc_train = 0
    total_loss_train = 0

    predicted_labels_train = []
    ground_truth_labels_train = []

    for train_input, train_label in tqdm(train_dataloader):
        train_input = train_input.to(device)
        train_label = train_label.to(device)

        output = model(train_input)

        batch_loss = criterion(output, train_label.long())
        total_loss_train += batch_loss.item()

        acc = (output.argmax(dim=1) == train_label).sum().item()
        total_acc_train += acc
        
        predicted_labels_train.extend(output.argmax(dim=1).cpu().numpy())
        ground_truth_labels_train.extend(train_label.cpu().numpy())

        model.zero_grad()
        batch_loss.backward()
        optimizer.step()

    total_acc_val = 0
    total_loss_val = 0

    predicted_labels_val = []
    ground_truth_labels_val = []

    with torch.no_grad(): #evaluation does not require changes in model
        for val_input, val_label in val_dataloader:
            output = model(val_input)

            batch_loss = criterion(output, val_label.long())
            total_loss_val += batch_loss.item()

            acc = (output.argmax(dim=1) == val_label).sum().item()
            total_acc_val += acc

            predicted_labels_val.extend(output.argmax(dim=1).cpu().numpy())
            ground_truth_labels_val.extend(val_label.cpu().numpy())

    train_acc.append(total_acc_train / len(df_train_x))
    train_loss.append(total_loss_train / len(df_train_x))
    test_acc.append(total_acc_val / len(df_test_x))
    test_loss.append(total_loss_val / len(df_test_x))

    conf_matrix_train = confusion_matrix(ground_truth_labels_train, predicted_labels_train)
    report_train = classification_report(ground_truth_labels_train, predicted_labels_train)

    conf_matrix_val = confusion_matrix(ground_truth_labels_val, predicted_labels_val)
    report_val = classification_report(ground_truth_labels_val, predicted_labels_val)

    print("\n\n----------------------------TRAIN----------------------------\n")
    print(conf_matrix_train)
    print(report_train)
    print()
    print("ground truth: " + str(ground_truth_labels_train))
    print("prediction:   " + str(predicted_labels_train))
    print()
    print("----------------------------TRAIN----------------------------\n\n")
    print("----------------------------TEST----------------------------\n")
    print(conf_matrix_val)
    print(report_val)
    print()
    print("ground truth: " + str(ground_truth_labels_val))
    print("prediction:   " + str(predicted_labels_val))
    print()
    print("----------------------------TEST----------------------------\n\n")

    print(f'Epochs: {epoch_num + 1} \
        | Train Loss: {total_loss_train / len(df_train_x): .3f} \
        | Train Accuracy: {total_acc_train / len(df_train_x): .3f} \
        | Val Loss: {total_loss_val / len(df_test_x): .3f} \
        | Val Accuracy: {total_acc_val / len(df_test_x): .3f}')

plot_statistics(train_acc, train_loss, test_acc, test_loss)

total_acc_test = 0

predicted_labels = []
ground_truth_labels = []

with torch.no_grad(): #testing does not require changes in model
    for test_input, test_label in val_dataloader:
        output = model(test_input)

        acc = (output.argmax(dim=1) == test_label).sum().item()
        total_acc_test += acc

        predicted_labels.extend(output.argmax(dim=1).cpu().numpy())
        ground_truth_labels.extend(test_label.cpu().numpy())

conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels)
report = classification_report(ground_truth_labels, predicted_labels)
    
print(conf_matrix)
print(report)

print("ground truth: " + str(ground_truth_labels))
print("prediction:   " + str(predicted_labels))
print(f'Test Accuracy: {(total_acc_test / len(df_test_x)): .3f}')
