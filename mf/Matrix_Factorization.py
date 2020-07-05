"""
想用pytorch实现下 Matrix Factorization
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import numpy as np
path = '../data/'
batch_size = 1024
device = torch.device('cuda:0')
learning_rate = 5e-4
weight_decay = 1e-5
epochs = 100


class MfDataset(Dataset):
    def __init__(self, u_id, i_id, rating):
        self.u_id = u_id
        self.i_id = i_id
        self.rating = rating

    def __getitem__(self, index):
        return self.u_id[index], self.i_id[index], self.rating[index]

    def __len__(self):
        return len(self.rating)


class MF(nn.Module):
    def __init__(self, num_users, num_items, mean, embedding_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)

        self.user_emb.weight.data.uniform_(0, 0.005)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        # 全局bias
        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        return (U * I).sum(1) + b_u + b_i + self.mean


def main():
    df = pd.read_csv(path + 'u.data', header=None, delimiter='\t')
    x, y = df.iloc[:, :2], df.iloc[:, 2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2020)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # 需要将数据全部转化为np.array, 否则后面的dataloader会报错， pytorch与numpy之间转换较好，与pandas转化容易出错
    train_dataset = MfDataset(np.array(x_train[0]), np.array(x_train[1]), np.array(y_train).astype(np.float32)) # 将标签设为np.float32类型， 否则会报错
    test_dataset = MfDataset(np.array(x_test[0]), np.array(x_test[1]), np.array(y_test).astype(np.float32))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


    mean_rating = df.iloc[:, 2].mean()
    num_users, num_items = max(df[0]) + 1, max(df[1]) + 1
    model = MF(num_users, num_items, mean_rating).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = torch.nn.MSELoss().to(device)

    for epoch in range(epochs):

        model.train()
        total_loss, total_len = 0, 0
        for x_u, x_i, y in train_dataloader:
            x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
            y_pre = model(x_u, x_i)
            loss = loss_func(y_pre, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_len += len(y)
        train_loss = total_loss / total_len

        model.eval()
        labels, predicts = [], []
        with torch.no_grad():
            for x_u, x_i, y in test_dataloader:
                x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
                y_pre = model(x_u, x_i)
                labels.extend(y.tolist())
                predicts.extend(y_pre.tolist())
        mse = mean_squared_error(np.array(labels), np.array(predicts))

        print("epoch {}, train loss is {}, val mse is {}".format(epoch, train_loss, mse))

if __name__ == '__main__':
    main()
