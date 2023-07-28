from utils.dataloader import get_bbc_tokenized_bert
from utils.dataloader import get_spam_tokenized_bert
import torch
from torch import nn
from transformers import BertModel
import numpy as np
from torch.utils.data import Dataset
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

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

#######CUSTOM DATASET FOR BERT CLASSIFIER#######
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

#######BERT CLASSIFIER INSTANCE#######
class BertClassifier(nn.Module):
    def __init__(self, dropout=0.1, target_classes=5, hidden_layers=768):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_layers, target_classes)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

def train(model, df_train_x, df_train_y, df_test_x, df_test_y, lr, epochs):
    train, val = Dataset(df_train_x, df_train_y), Dataset(df_test_x, df_test_y)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True) #add randomness to training data
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=32)

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

            predicted_labels_train = []
            ground_truth_labels_train = []
        
            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

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
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

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

def evaluate(model, df_test_x, df_test_y):
    test = Dataset(df_test_x, df_test_y)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    predicted_labels = []
    ground_truth_labels = []

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

            predicted_labels.extend(output.argmax(dim=1).cpu().numpy())
            ground_truth_labels.extend(test_label.cpu().numpy())

    conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels)
    report = classification_report(ground_truth_labels, predicted_labels)
		
    print(conf_matrix)
    print(report)
    
    print("ground truth: " + str(ground_truth_labels))
    print("prediction:   " + str(predicted_labels))
    print(f'Test Accuracy: {total_acc_test / len(df_test_x): .3f}')
    

import argparse

parser = argparse.ArgumentParser(description='BERT based deeplearning.')

parser.add_argument('-a', '--augmentation', type=bool, default=False,
                    help='Choose whether data augmentation should be performed before training.')
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5,
                    help='Choose learning rate of the BERT model.')
parser.add_argument('-e', '--epochs', type=int, default=5,
                    help='Choose epochs of the BERT model.')
parser.add_argument('-s', '--hidden_layers', type=int, default=768,
                    help='Choose hidden layers of the BERT model.')
parser.add_argument('-d', '--dropout', type=float, default=0.1,
                    help='Choose dropout rate of the BERT model.') 
parser.add_argument('dataset', type=str, help='"bbc" | "spam"')

# Parse the command-line arguments
args = parser.parse_args()

augmentation = args.augmentation
epochs = args.epochs
lr = args.learning_rate
dataset = args.dataset
dropout = args.dropout
hidden_layers = args.hidden_layers

if dataset == "bbc":
    model = BertClassifier(target_classes=5, dropout=dropout, hidden_layers=hidden_layers)
    df_train_x, df_train_y, df_test_x, df_test_y = get_bbc_tokenized_bert(False, False, augmentation)
    train(model, df_train_x, df_train_y, df_test_x, df_test_y, lr, epochs)
    evaluate(model, df_test_x, df_test_y)
elif dataset == "spam":
    model = BertClassifier(target_classes=2, dropout=dropout, hidden_layers=hidden_layers)
    df_train_x, df_train_y, df_test_x, df_test_y = get_spam_tokenized_bert(False, False, augmentation)
    train(model, df_train_x, df_train_y, df_test_x, df_test_y, lr, epochs)
    evaluate(model, df_test_x, df_test_y)
else:
    print("\t\tUnknown dataset, choose either bbc or spam.")
