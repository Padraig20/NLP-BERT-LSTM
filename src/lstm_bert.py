import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from utils.dataloader import get_bbc_tokenized_bert
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm

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
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_sample = self.data[index]
        label = self.labels[index]
        return data_sample, label

class LSTMWithBertEmbeddings(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(LSTMWithBertEmbeddings, self).__init__()
        self.bert = BertModel.from_pretrained('distilbert-base-uncased')
        self.embedding_dim = self.bert.config.hidden_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # LSTM layer
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        # BERT embeddings
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]

        # LSTM layer
        lstm_output, _ = self.lstm(bert_output)

        # Final output for classification
        output = self.fc(lstm_output[:, -1, :])  # Use the last output from the LSTM
        return output

def train(model, df_train_x, df_train_y, df_test_x, df_test_y, lr, epochs):

    train, val = Dataset(df_train_x, df_train_y), Dataset(df_test_x, df_test_y)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True) #add randomness to training data
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= lr)

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
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
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

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_val += acc

        train_acc.append(total_acc_train / len(df_train_x))
        train_loss.append(total_loss_train / len(df_train_x))
        test_acc.append(total_acc_val / len(df_test_x))
        test_loss.append(total_loss_val / len(df_test_x))

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

parser = argparse.ArgumentParser(description='BERT+LSTM based deeplearning.')

parser.add_argument('-a', '--augmentation', type=bool, default=False,
                    help='Choose whether data augmentation should be performed before training.')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-6,
                    help='Choose learning rate of the LSTM model.')
parser.add_argument('-e', '--epochs', type=int, default=5,
                    help='Choose epochs of the LSTM model.')
parser.add_argument('-l', '--hidden_layers', type=int, default=5,
                    help='Choose hidden layers of the LSTM model.')
parser.add_argument('dataset', type=str, help='"bbc" | ""')

# Parse the command-line arguments
args = parser.parse_args()

augmentation = args.augmentation
epochs = args.epochs
lr = args.learning_rate
dataset = args.dataset
hidden_layers = args.hidden_layers

if dataset == "bbc":
    df_train_x, df_train_y, df_test_x, df_test_y = get_bbc_tokenized_bert(False, False, augmentation)
    df_train_y = torch.tensor(df_train_y)
    df_test_y = torch.tensor(df_test_y)

    model = LSTMWithBertEmbeddings(hidden_dim=hidden_layers, output_dim=5)  # Adjust output_dim based on your task

    train(model, df_train_x, df_train_y, df_test_x, df_test_y, lr, epochs)

    evaluate(model, df_test_x, df_train_y)

else: 
    print("Unknown Dataset, choose either bbc or.")

