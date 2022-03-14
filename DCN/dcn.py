import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    def __init__(self, inputs_dim, hidden_units, dropout_rate):
        super(DNN, self).__init__()
        self.inputs_dim = inputs_dim
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout_rate)

        self.hidden_units = [inputs_dim] + list(self.hidden_units)
        self.linear = nn.ModuleList([
            nn.Linear(self.hidden_units[i], self.hidden_units[i+1]) for i in range(len(self.hidden_units)-1)
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

class CrossNet(nn.Module):
    def __init__(self, in_features, layer_num=2, parameterization='vector', seed=2022):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        elif self.parameterization == 'matrix':
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[0])

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_1 = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                x1_w = torch.tensordot(x_1, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, x1_w)
                x_1 = dot_ + self.bias[i] + x_1
            else:
                x1_w = torch.tensordot(self.kernels[i], x_1)
                dot_ = x1_w + self.bias[i]
                x_1 = x_0 * dot_ + x_1
        x_1 = torch.squeeze(x_1, dim=2)
        return x_1

class DCN(nn.Module):
    def __init__(self, feat_size, embedding_size, linear_feature_columns, dnn_feature_columns, cross_num=2,
                 cross_param='vector', dnn_hidden_units=(128, 128,), init_std=0.0001, seed=2022, l2_reg=0.00001,
                 drop_rate=0.5):
        super(DCN, self).__init__()
        self.feat_size = feat_size
        self.embedding_size = embedding_size
        self.dnn_hidden_units = dnn_hidden_units
        self.cross_num = 2
        self.cross_param = cross_param
        self.drop_rate = drop_rate
        self.l2_reg = 0.00001

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

        self.dense_feature_columns = list(filter(lambda x:x[1]=='dense', dnn_feature_columns))
        self.sparse_feature_columns = list(filter(lambda x:x[1]=='sparse', dnn_feature_columns))

        self.embedding_dic = nn.ModuleDict({feat[0]:nn.Embedding(feat_size[feat[0]], self.embedding_size, sparse=False)
                                            for feat in self.sparse_feature_columns})

        self.feature_index = defaultdict(int)
        start = 0
        for feat in self.feat_size:
            self.feature_index[feat] = start
            start += 1


        inputs_dim = len(self.dense_feature_columns)+self.embedding_size*len(self.sparse_feature_columns)

        self.dnn = DNN(inputs_dim,self.dnn_hidden_units, 0.5)

        self.crossnet = CrossNet(inputs_dim, layer_num=self.cross_num, parameterization=self.cross_param)
        self.dnn_linear = nn.Linear(inputs_dim+dnn_hidden_units[-1], 1, bias=False)

        dnn_hidden_units = [len(feat_size)] + list(dnn_hidden_units) + [1]
        self.linear = nn.ModuleList([
            nn.Linear(dnn_hidden_units[i], dnn_hidden_units[i+1]) for i in range(len(dnn_hidden_units)-1)
        ])
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, X):

        logit = X
        for i in range(len(self.linear)):
            fc = self.linear[i](logit)
            fc = self.act(fc)
            fc = self.dropout(fc)
            logit = fc

        sparse_embedding = [self.embedding_dic[feat[0]](X[:, self.feature_index[feat[0]]].long()).reshape(X.shape[0], 1, -1)
                            for feat in self.sparse_feature_columns]
        dense_values = [X[:, self.feature_index[feat[0]]].reshape(-1, 1) for feat in self.dense_feature_columns]

        dense_input = torch.cat(dense_values, dim=1)
        sparse_input = torch.cat(sparse_embedding, dim=1)

        # 拉直 本来是 [batch_size, sparase特征数, 嵌入维度] =》 [batch_size, sparase特征数 * 嵌入维度]
        sparse_input = torch.flatten(sparse_input, start_dim=1)

        dnn_input = torch.cat((dense_input, sparse_input), dim=1)

        # print('sparse input size', sparse_input.shape)
        # print('dense input size', dense_input.shape)
        # print('dnn input size', dnn_input.shape)

        deep_out = self.dnn(dnn_input)
        cross_out = self.crossnet(dnn_input)
        stack_out = torch.cat((cross_out, deep_out), dim=-1)

        logit += self.dnn_linear(stack_out)
        #print('logit size', logit.shape)
        y_pred = torch.sigmoid(logit)
        #print('y_pred', y_pred.shape)
        return y_pred


if __name__ == '__main__':

    batch_size = 1024
    lr = 1e-2
    wd = 1e-3
    epoches = 20
    seed = 2022
    embedding_size = 4
    device = 'cuda:0'

    sparse_feature = ['C' + str(i) for i in range(1, 27)]
    dense_feature = ['I' + str(i) for i in range(1, 14)]
    col_names = ['label'] + dense_feature + sparse_feature
    data = pd.read_csv('./data/dac_sample.txt', names=col_names, sep='\t')

    data[sparse_feature] = data[sparse_feature].fillna('-1', )
    data[dense_feature] = data[dense_feature].fillna('0',)
    target = ['label']

    feat_sizes = {}
    feat_sizes_dense = {feat:1 for feat in dense_feature}
    feat_sizes_sparse = {feat:len(data[feat].unique()) for feat in sparse_feature}
    feat_sizes.update(feat_sizes_dense)
    feat_sizes.update(feat_sizes_sparse)

    for feat in sparse_feature:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    nms = MinMaxScaler(feature_range=(0, 1))
    data[dense_feature] = nms.fit_transform(data[dense_feature])

    fixlen_feature_columns = [(feat,'sparse') for feat in sparse_feature ]  + [(feat,'dense') for feat in dense_feature]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    train, test = train_test_split(data, test_size=0.2, random_state=seed)

    device = 'cuda:0'
    model = DCN(feat_sizes, embedding_size, linear_feature_columns, dnn_feature_columns).to(device)

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


