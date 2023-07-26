from collections import Counter
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
import numpy as np
from utils.dataloader import get_bbc_vanilla
from utils.dataloader import get_bbc_tokenized_bert
from utils.dataloader import get_bbc_tokenized_torch
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

#######CUSTOM DATASET FOR LSTM CLASSIFIER#######
class Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print(idx)
        print(self.data[idx])
        print(self.labels[idx])
        return self.data[idx], self.labels[idx]

#######LSTM CLASSIFIER INSTANCE#######
#class LSTMModel(nn.Module):
#    def __init__(self, input_size, hidden_size, num_layers, output_size):
#        super(LSTMModel, self).__init__()
#        self.hidden_size = hidden_size
#        self.num_layers = num_layers

#        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#        self.fc = nn.Linear(hidden_size, output_size)

#    def forward(self, x):
#        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

#        out, _ = self.lstm(x, (h0, c0))
#        out = self.fc(out[:, -1, :])
#        return out

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        packed_output, (hidden, cell) = self.lstm(text)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        return self.fc(output[-1, :, :])


def train_bert(model, embeddings_train, df_train_y, embeddings_test, df_test_y, lr, epochs):

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
            output = model(embeddings_train)

            train_labels_tensor = torch.tensor(df_train_y.values, dtype=torch.long)

            loss_train = criterion(output, train_labels_tensor)

            # Calculate accuracy on the training set
            _, predicted_classes = torch.max(output, 1)
            correct = (predicted_classes == train_labels_tensor).sum().item()
            total = train_labels_tensor.size(0)
            accuracy_train = correct / total

            model.zero_grad()
            loss_train.backward()
            optimizer.step()

            with torch.no_grad(): #evaluation does not require changes in model
                output = model(embeddings_test)

                test_labels_tensor = torch.tensor(df_test_y.values, dtype=torch.long)

                loss_val = criterion(output, test_labels_tensor).item()

                # Calculate accuracy on the training set
                _, predicted_classes = torch.max(output, 1)
                correct = (predicted_classes == test_labels_tensor).sum().item()
                total = test_labels_tensor.size(0)
                accuracy_val = correct / total

            train_acc.append(accuracy_train)
            train_loss.append(loss_train.item())
            test_acc.append(accuracy_val)
            test_loss.append(loss_val)

            print(f'Epochs: {epoch_num + 1} \
                | Train Loss: {loss_train.item(): .3f} \
                | Train Accuracy: {accuracy_train: .3f} \
                | Val Loss: {loss_val: .3f} \
                | Val Accuracy: {accuracy_val: .3f}')

    plot_statistics(train_acc, train_loss, test_acc, test_loss)


def evaluate_bert(model, embeddings_test, df_test_y):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad(): #testing does not require changes in model
        output = model(embeddings_test)

        test_labels_tensor = torch.tensor(df_test_y.values, dtype=torch.long)

        # Calculate accuracy on the training set
        _, predicted_classes = torch.max(output, 1)
        correct = (predicted_classes == test_labels_tensor).sum().item()
        total = test_labels_tensor.size(0)
        accuracy_val = correct / total
    
    print(f'Test Accuracy: {accuracy_val: .3f}')


def train(model, df_train_x, df_train_y, df_test_x, df_test_y, lr, epochs):

    train = Dataset(df_train_x, df_train_y.values)
    test = Dataset(df_test_x, df_test_y.values)

    train_dataloader = DataLoader(train, batch_size=2) #add randomness to training data
    val_dataloader = DataLoader(test, batch_size=2)

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

            for train_input, train_label in tqdm(train_dataloader):
                output = model(train_input)

                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad(): #evaluation does not require changes in model
                for val_input, val_label in val_dataloader:
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()

                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

            train_acc.append(total_acc_train / len(df_train_x))
            train_loss.append(total_loss_train / len(df_train_x))
            test_acc.append(total_loss_val / len(df_test_x))
            test_loss.append(total_acc_val / len(df_test_x))

            print(f'Epochs: {epoch_num + 1} \
                | Train Loss: {total_loss_train / len(df_train_x): .3f} \
                | Train Accuracy: {total_acc_train / len(df_train_x): .3f} \
                | Val Loss: {total_loss_val / len(df_test_x): .3f} \
                | Val Accuracy: {total_acc_val / len(df_test_x): .3f}')

    plot_statistics(train_acc, train_loss, test_acc, test_loss)

def evaluate(model, df_test_x, df_test_y):
    test = Dataset(df_test_x, df_test_y)

    test_dataloader = DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad(): #testing does not require changes in model
        for test_input, test_label in test_dataloader:
              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
    
    print(f'Test Accuracy: {total_acc_test / len(df_test_x): .3f}')


import argparse

parser = argparse.ArgumentParser(description='LSTM based deeplearning.')

parser.add_argument('-a', '--augmentation', type=bool, default=False,
                    help='Choose whether data augmentation should be performed before training.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                    help='Choose learning rate of the LSTM model.')
parser.add_argument('-s', '--hidden_size', type=int, default=5,
                    help='Choose hidden size of the LSTM model.')
parser.add_argument('-l', '--layers', type=int, default=5,
                    help='Choose number of layers of the LSTM model.')
parser.add_argument('-e', '--epochs', type=int, default=5,
                    help='Choose epochs of the LSTM model.')
parser.add_argument('dataset', type=str, help='"bbc" | "bbc-bert" | ""')

args = parser.parse_args()

lr = args.learning_rate
augmentation = args.augmentation
hidden_size = args.hidden_size
num_layers = args.layers
epochs = args.epochs
dataset = args.dataset

if dataset == "bbc-bert":
    embeddings_train, df_train_y, embeddings_test, df_test_y = get_bbc_tokenized_bert(False, True, augmentation)

    input_size = embeddings_train.size(-1) #vocabulary size
    output_size = 5 # the number of output classes

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    train_bert(model, embeddings_train, df_train_y, embeddings_test, df_test_y, lr, epochs)

    evaluate_bert(model, embeddings_test, df_test_y)
elif dataset == "bbc":
    import torchtext
    from torch.nn.utils.rnn import pad_sequence

    df_train_x, df_train_y, df_test_x, df_test_y = get_bbc_tokenized_torch(False, augmentation)

    print(df_train_x)

    # get vocabulary
    vocab_train = torchtext.vocab.build_vocab_from_iterator(df_train_x)
    vocab_test = torchtext.vocab.build_vocab_from_iterator(df_test_x)

    # Calculate the correct input size (number of unique words in the vocabulary)
    input_size = len(vocab_train)

    # turn tokens into word indices
    df_train_x = [[vocab_train[token] for token in tokens] for tokens in df_train_x]
    df_test_x = [[vocab_test[token] for token in tokens] for tokens in df_test_x]

    # pad sequences for tensors
    df_train_x = pad_sequence([torch.tensor(seq) for seq in df_train_x], batch_first=True)
    df_test_x = pad_sequence([torch.tensor(seq) for seq in df_test_x], batch_first=True)

    output_size = 5 # the number of output classes

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    train(model, df_train_x, df_train_y, df_test_x, df_test_y, lr, epochs)

    #evaluate(model, df_train_x, df_test_y)
else:
    print("\t\tUnknown dataset, choose either bbc or .")
