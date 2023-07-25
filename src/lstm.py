from collections import Counter
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from utils.dataloader import get_bbc_vanilla
from utils.dataloader import get_bbc_tokenized_bert
import torch
from torch import nn
from transformers import BertModel
import numpy as np
from torch.utils.data import Dataset
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

#######LSTM CLASSIFIER INSTANCE#######
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train(model, embeddings_train, df_train_y, embeddings_test, df_test_y, lr, epochs):

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


def evaluate(model, embeddings_test, df_test_y):

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
parser.add_argument('dataset', type=str, help='"bbc" | ""')

args = parser.parse_args()

lr = args.learning_rate
augmentation = args.augmentation
hidden_size = args.hidden_size
num_layers = args.layers
epochs = args.epochs
dataset = args.dataset

if dataset == "bbc":
    embeddings_train, df_train_y, embeddings_test, df_test_y = get_bbc_tokenized_bert(False, True, augmentation)

    input_size = embeddings_train.size(-1) #vocabulary size
    output_size = 5 # the number of output classes

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    train(model, embeddings_train, df_train_y, embeddings_test, df_test_y, lr, epochs)

    evaluate(model, embeddings_test, df_test_y)
else:
    print("\t\tUnknown dataset, choose either bbc or .")
