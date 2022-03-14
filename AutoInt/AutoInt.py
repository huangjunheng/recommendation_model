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

        # self.bn = nn.ModuleList([
        #     nn.Linear(self.hidden_units[i], self.hidden_units[i + 1]) for i in range(len(self.hidden_units) - 1)
        # ])
        self.activation = nn.ReLU()

    def forward(self, X):
        inputs = X
        for i in range(len(self.linear)):
            fc = self.linear[i](inputs)
            fc = self.activation(fc)
            fc = self.dropout(fc)
            inputs = fc
        return inputs


class InteractingLayer(nn.Module):
    def __init__(self, embedding_size, head_num=2, use_res=True, scaling=False):
        super(InteractingLayer, self).__init__()
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling

        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))

        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

    def forward(self, inputs):

        # inputs: [1024, 26, 4]
        #keys: [1024, 26, 4]
        querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_Key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        # keys: [2, 1024, 26, 2]
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        # inner_product: [2, 1024, 26, 26]
        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)

        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = F.softmax(inner_product, dim=-1)

        # [2, 1024, 26, 2]
        result = torch.matmul(self.normalized_att_scores, values)
        # [1, 1024, 26, 4]
        result = torch.cat(torch.split(result, 1, ), dim=-1)
        # [1024, 26, 4]
        result = torch.squeeze(result, dim=0)
        if self.use_res:
            result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))
        result = F.relu(result)
        return result

class AutoInt(nn.Module):
    def __init__(self, feat_size, embedding_size, linear_feature_columns, dnn_feature_columns, att_layer_num=3,
                 att_head_num=2,
                 att_res=True, dnn_hidden_units=(256, 128)):
        super(AutoInt, self).__init__()
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

        self.dnn = DNN(len(self.dense_feature_columns) + embedding_size * len(self.embedding_dic), dnn_hidden_units,
                       0.5)

        self.dnn_linear = nn.Linear(dnn_hidden_units[-1] + embedding_size * len(self.embedding_dic), 1, bias=False)

        for name, tensor in self.dnn_linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.00001)

        dnn_hidden_units = [len(feat_size), 1]
        self.linear = nn.ModuleList([
            nn.Linear(dnn_hidden_units[i], dnn_hidden_units[i + 1]) for i in range(len(dnn_hidden_units) - 1)
        ])
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.00001)

        self.int_layers = nn.ModuleList([
            InteractingLayer(embedding_size, att_head_num, att_res) for _ in range(att_layer_num)
        ])

        self.out = nn.Sigmoid()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        # X [1024, 39]
        logit = X
        for i in range(len(self.linear)):
            fc = self.linear[i](logit)
            fc = self.act(fc)
            fc = self.dropout(fc)
            logit = fc
        # logit [1024, 1]
        sparse_embedding = [
            self.embedding_dic[feat[0]](X[:, self.feature_index[feat[0]]].long()).reshape(X.shape[0], 1, -1)
            for feat in self.sparse_feature_columns]
        sparse_input = torch.cat(sparse_embedding, dim=1)
        sparse_input = torch.flatten(sparse_input, start_dim=1)
        dense_values = [X[:, self.feature_index[feat[0]]].reshape(-1, 1) for feat in self.dense_feature_columns]
        dense_input = torch.cat(dense_values, dim=1)

        # att_input [1024, 26, 4]
        att_input = torch.cat(sparse_embedding, dim=1)
        for layer in self.int_layers:
            att_input = layer(att_input)
        # att_out [1024, 104]
        att_output = torch.flatten(att_input, start_dim=1)
        #print('att_output shape', att_output.shape)

        # dnn_input [1024, 117] 26*4+13
        dnn_input = torch.cat((dense_input, sparse_input), dim=1)
        # deep_out [1024, 128]
        deep_out = self.dnn(dnn_input)
        stack_out = torch.cat((att_output, deep_out), dim=-1)
        #print('stack_out shape', stack_out.shape)
        logit += self.dnn_linear(stack_out)
        #print('logit shape', logit.shape)
        y_pred = torch.sigmoid(logit)
        return y_pred


if __name__ == '__main__':

    batch_size = 1024
    lr = 1e-3
    wd = 0
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
    model = AutoInt(feat_sizes, embedding_size, linear_feature_columns, dnn_feature_columns).to(device)

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
