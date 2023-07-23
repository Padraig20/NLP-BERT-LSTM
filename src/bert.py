from utils.dataloader import get_bbc_tokenized_bert
import torch
from torch import nn
from transformers import BertModel
import numpy as np
from transformers import BertTokenizer

from torch.utils.data import Dataset

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

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

from torch.optim import Adam
from tqdm import tqdm

def train(model, df_train_x, df_train_y, df_test_x, df_test_y, lr, epochs):

    train, val = Dataset(df_train_x, df_train_y), Dataset(df_test_x, df_test_y)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= lr)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                #print(train_input)
                #print(train_label)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
                print("mask:")
                print(mask.shape)
                print("input_id")
                print(input_id.shape)

                output = model(input_id, mask)

                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()

                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(df_train_x): .3f} \
                | Train Accuracy: {total_acc_train / len(df_train_x): .3f} \
                | Val Loss: {total_loss_val / len(df_test_x): .3f} \
                | Val Accuracy: {total_acc_val / len(df_test_x): .3f}')

def evaluate(model, df_test_x, df_test_y):

    test = Dataset(df_test_x, df_test_y)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
    
    print(f'Test Accuracy: {total_acc_test / len(df_test_x): .3f}')
    

EPOCHS = 5
model = BertClassifier()
LR = 1e-6

df_train_x, df_train_y, df_test_x, df_test_y = get_bbc_tokenized_bert(False)

train(model, df_train_x, df_train_y, df_test_x, df_test_y, LR, EPOCHS)

evaluate(model, df_test_x, df_test_y)
