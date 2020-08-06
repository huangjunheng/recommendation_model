'''
pytorch实现因子分解机模型
author: huangjunheng
date: 2020/8/6
'''
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

learning_rate = 1e-3
weight_decay = 1e-4
epochs = 100
batch_szie = 1024
min_val, max_val = 1.0, 5.0
device = torch.device('cuda:0')
id_embedding_dim = 256  # id嵌入向量的长度


# fm模型
class FmLayer(nn.Module):

    def __init__(self, p, k):
        super(FmLayer, self).__init__()
        self.p, self.k = p, k
        self.linear = nn.Linear(self.p, 1, bias=True)
        self.v = nn.Parameter(torch.Tensor(self.p, self.k), requires_grad=True)
        self.v.data.uniform_(-0.01, 0.01)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        linear_part = self.linear(x)
        inter_part1 = torch.pow(torch.mm(x, self.v), 2)
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        pair_interactions = torch.sum(torch.sub(inter_part1, inter_part2), dim=1)
        self.drop(pair_interactions)
        output = linear_part.transpose(1, 0) + 0.5 * pair_interactions
        return output.view(-1, 1)


class FM(nn.Module):
    def __init__(self, user_nums, item_nums, id_embedding_dim):
        super(FM, self).__init__()
        # 对用户， 物品的id进行嵌入
        self.user_id_vec = nn.Embedding(user_nums, id_embedding_dim)
        self.item_id_vec = nn.Embedding(item_nums, id_embedding_dim)

        self.fm = FmLayer(id_embedding_dim * 2, 10)

    def forward(self, u_id, i_id):
        u_vec = self.user_id_vec(u_id)
        i_vec = self.item_id_vec(i_id)
        x = torch.cat((u_vec, i_vec), dim=1)
        rate = self.fm(x)
        return rate


class FmDataset(Dataset):
    def __init__(self, uid, iid, rating):
        self.uid = uid
        self.iid = iid
        self.rating = rating

    def __getitem__(self, index):
        return self.uid[index], self.iid[index], self.rating[index]

    def __len__(self):
        return len(self.uid)


def train_iter(model, optimizer, data_loder, criterion):
    model.train()
    total_loss = 0
    total_len = 0

    for index, (x_u, x_i, y) in enumerate(data_loder):
        x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
        y = (y - min_val) / (max_val - min_val) + 0.01
        y_pre = model(x_u, x_i)

        loss = criterion(y.view(-1, 1), y_pre)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_pre)
        total_len += len(y_pre)

    loss = total_loss / total_len
    return loss


def val_iter(model, data_loader):
    model.eval()
    labels, predicts = list(), list()

    with torch.no_grad():
        for x_u, x_i, y in data_loader:
            x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
            y_pre = model(x_u, x_i)
            y_pre = min_val + (y_pre - 0.01) * (max_val - min_val)
            y_pre = torch.where(y_pre > 5.0, torch.full_like(y_pre, 5.0), y_pre)
            y_pre = torch.where(y_pre < 1.0, torch.full_like(y_pre, 1.0), y_pre)
            labels.extend(y.tolist())
            predicts.extend(y_pre.tolist())
    mse = mean_squared_error(np.array(labels), np.array(predicts))

    return mse


def main():
    df = pd.read_csv('u.data', header=None, delimiter='\t')
    len_df, u_max_id, i_max_id = len(df), max(df[0]) + 1, max(df[1]) + 1
    print(df.shape, max(df[0]), max(df[1]))
    x, y = df.iloc[:, :2], df.iloc[:, 2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2020)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=2020)
    train_loader = DataLoader(
        FmDataset(np.array(x_train[0]), np.array(x_train[1]), np.array(y_train).astype(np.float32)), batch_size=batch_szie)
    val_loader = DataLoader(FmDataset(np.array(x_val[0]), np.array(x_val[1]), np.array(y_val).astype(np.float32)), batch_size=batch_szie)
    test_loader = DataLoader(FmDataset(np.array(x_test[0]), np.array(x_test[1]), np.array(y_test).astype(np.float32)), batch_size=batch_szie)

    # 模型初始化
    model = FM(u_max_id, i_max_id, id_embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = torch.nn.MSELoss().to(device)

    # 训练模型
    best_val_mse, best_val_epoch = 10, 0
    for epoch in range(epochs):
        loss = train_iter(model, optimizer, train_loader, loss_func)
        mse = val_iter(model, val_loader)
        print("epoch:{}, loss:{:.5}, mse:{:.5}".format(epoch, loss, mse))
        if best_val_mse > mse:
            best_val_mse, best_val_epoch = mse, epoch
            torch.save(model, 'best_model')
    print("best val epoch is {}, mse is {}".format(best_val_epoch, best_val_mse))
    model = torch.load('best_model').to(device)
    test_mse = val_iter(model, test_loader)
    print("test mse is {}".format(test_mse))


if __name__ == '__main__':
    main()
