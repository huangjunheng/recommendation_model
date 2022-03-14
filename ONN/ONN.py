import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import OrderedDict, namedtuple, defaultdict


def get_auc(loader, model):
    pred, target = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device).float()
            y_hat = model(x)
            pred += list(y_hat.cpu().numpy())
            target += list(y.cpu().numpy())
    auc = roc_auc_score(target, pred)
    return auc


class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, dropout_rate, ):
        super(DNN, self).__init__()
        self.inputs_dim = inputs_dim
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout_rate)

        self.hidden_units = [inputs_dim] + list(self.hidden_units)
        self.linear = nn.ModuleList([
            nn.Linear(self.hidden_units[i], self.hidden_units[i + 1]) for i in range(len(self.hidden_units) - 1)
        ])
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)

        self.activation = nn.ReLU()

    def forward(self, X):
        inputs = X
        for i in range(len(self.linear)):
            fc = self.linear[i](inputs)
            fc = self.activation(fc)
            fc = self.dropout(fc)
            inputs = fc
        return inputs

class Interac(nn.Module):
    def __init__(self, first_size, second_size, emb_size):
        super(Interac, self).__init__()
        self.emb1 = nn.Embedding(first_size, emb_size)
        self.emb2 = nn.Embedding(second_size, emb_size)
        nn.init.normal_(self.emb1.weight, mean=0, std=0.00001)

    def forward(self, first, second):
        frist_emb = self.emb1(first)
        second_emb = self.emb2(second)
        y = frist_emb * second_emb
        return y

class ONN(nn.Module):
    def __init__(self, feat_size, embedding_size, linear_feature_columns, dnn_feature_columns,
                 dnn_hidden_units=(128, 128), l2_reg=1e-5, dnn_dropout=0.5):
        super(ONN, self).__init__()
        self.sparse_feature_columns = list(filter(lambda x: x[1] == 'sparse', dnn_feature_columns))
        self.embedding_dic = nn.ModuleDict({
            feat[0]: nn.Embedding(feat_size[feat[0]], embedding_size, sparse=False) for feat in
            self.sparse_feature_columns
        })
        self.dense_feature_columns = list(filter(lambda x: x[1] == 'dense', dnn_feature_columns))

        self.feature_index = defaultdict(int)
        start = 0
        for feat in feat_size:
            self.feature_index[feat] = start
            start += 1

        self.second_order_embedding_dic = self.__create_second_order_embedding_matrix(feat_size, embedding_size)

        dim = self.__compute_nffm_dnn_dim(embedding_size)
        self.dnn = DNN(int(dim), dnn_hidden_units, 0.5)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)

        dnn_hidden_units = [len(feat_size), 128, 1]
        self.linear = nn.ModuleList([
            nn.Linear(dnn_hidden_units[i], dnn_hidden_units[i + 1]) for i in range(len(dnn_hidden_units) - 1)
        ])
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.00001)

        self.out = nn.Sigmoid()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dnn_dropout)

    def __create_second_order_embedding_matrix(self, feat_size, embedding_size, init_std=0.0001):
        temp_dic = {}
        for first_index in range(len(self.sparse_feature_columns)-1):
            for second_index in range(first_index+1, len(self.sparse_feature_columns)):
                first_name = self.sparse_feature_columns[first_index][0]
                second_name = self.sparse_feature_columns[second_index][0]
                temp_dic[first_name + '+' + second_name] = Interac(feat_size[first_name],
                                                                    feat_size[second_name], embedding_size)
        return nn.ModuleDict(temp_dic)

    def __compute_nffm_dnn_dim(self, embedding_size):
        x, y = len(self.sparse_feature_columns), len(self.dense_feature_columns)
        return x*(x-1)/2*embedding_size + y

    def __input_from_second_order_column(self, X):
        second_order_embedding_list = []
        for first_index in range(len(self.sparse_feature_columns)-1):
            for second_index in range(first_index+1, len(self.sparse_feature_columns)):
                first_name = self.sparse_feature_columns[first_index][0]
                second_name = self.sparse_feature_columns[second_index][0]
                second_order_embedding_list.append(
                    self.second_order_embedding_dic[first_name+'+'+second_name](
                        X[:, self.feature_index[first_name]].reshape(-1, 1).long(),
                        X[:, self.feature_index[second_name]].reshape(-1, 1).long()
                    )
                )
        return second_order_embedding_list

    def forward(self, X):
        logit = X
        for i in range(len(self.linear)):
            fc = self.linear[i](logit)
            fc = self.act(fc)
            fc = self.dropout(fc)
            logit = fc
        # logit [1024, 1]
        dense_values = [X[:, self.feature_index[feat[0]]].reshape(-1, 1) for feat in self.dense_feature_columns]
        dense_input = torch.cat(dense_values, dim=1)
        sparse_embedding = self.__input_from_second_order_column(X) # list 325
        sparse_input = torch.cat(sparse_embedding, dim=1)  # [1024, 325, 4]
        sparse_input = torch.flatten(sparse_input, start_dim=1) # [1024, 1300]
        dnn_input = torch.cat((dense_input, sparse_input), dim=1) # [1024, 1313]
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        final_logit = dnn_logit + logit

        y_pred = torch.sigmoid(final_logit)
        return y_pred


if __name__ == '__main__':

    batch_size = 1024
    lr = 1e-4
    wd = 1e-5
    epoches = 100
    seed = 2022
    embedding_size = 4
    device = 'cuda:0'

    sparse_feature = ['C' + str(i) for i in range(1, 27)]
    dense_feature = ['I' + str(i) for i in range(1, 14)]
    col_names = ['label'] + dense_feature + sparse_feature
    data = pd.read_csv('./data/dac_sample.txt', names=col_names, sep='\t')

    data[sparse_feature] = data[sparse_feature].fillna('-1', )
    data[dense_feature] = data[dense_feature].fillna('0', )
    target = ['label']

    feat_sizes = {}
    feat_sizes_dense = {feat: 1 for feat in dense_feature}
    feat_sizes_sparse = {feat: len(data[feat].unique()) for feat in sparse_feature}
    feat_sizes.update(feat_sizes_dense)
    feat_sizes.update(feat_sizes_sparse)

    for feat in sparse_feature:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    nms = MinMaxScaler(feature_range=(0, 1))
    data[dense_feature] = nms.fit_transform(data[dense_feature])

    fixlen_feature_columns = [(feat, 'sparse') for feat in sparse_feature] + [(feat, 'dense') for feat in dense_feature]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    train, test = train_test_split(data, test_size=0.2, random_state=seed)

    device = 'cuda:0'
    model = ONN(feat_sizes, embedding_size, linear_feature_columns, dnn_feature_columns).to(device)

    train_label = pd.DataFrame(train['label'])
    train = train.drop(columns=['label'])
    train_tensor_data = TensorDataset(torch.from_numpy(np.array(train)), torch.from_numpy(np.array(train_label)))
    train_loader = DataLoader(train_tensor_data, shuffle=True, batch_size=batch_size)

    test_label = pd.DataFrame(test['label'])
    test = test.drop(columns=['label'])
    test_tensor_data = TensorDataset(torch.from_numpy(np.array(test)), torch.from_numpy(np.array(test_label)))
    test_loader = DataLoader(test_tensor_data, batch_size=batch_size)

    loss_func = nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epoches):
        total_loss_epoch = 0.0
        total_tmp = 0
        model.train()
        for index, (x, y) in enumerate(train_loader):
            x, y = x.to(device).float(), y.to(device).float()
            y_hat = model(x)

            optimizer.zero_grad()
            loss = loss_func(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss_epoch += loss.item()
            total_tmp += 1
        auc = get_auc(test_loader, model)
        print('epoch/epoches: {}/{}, train loss: {:.3f}, test auc: {:.3f}'.format(epoch, epoches,
                                                                                  total_loss_epoch / total_tmp, auc))
