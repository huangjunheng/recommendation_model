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

class FM(nn.Module):
    def __init__(self):
        super(FM, self).__init__()
    def forward(self, inputs):
        fm_input = inputs
        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
        return cross_term

class AFMLayer(nn.Module):
    def __init__(self, embedding_size, attention_factor=4, l2_reg=0.0, drop_rate=0.0):
        super(AFMLayer, self).__init__()

        self.embedding_size = embedding_size
        self.attention_factor = attention_factor
        self.l2_reg = l2_reg
        self.drop_rate = drop_rate

        self.attention_W = nn.Parameter(torch.Tensor(self.embedding_size, self.attention_factor))
        self.attention_b = nn.Parameter(torch.Tensor(self.attention_factor))
        self.projection_h = nn.Parameter(torch.Tensor(self.attention_factor, 1))
        self.projection_p = nn.Parameter(torch.Tensor(self.embedding_size, 1))

        for tensor in [self.attention_W, self.projection_h, self.projection_p]:
            nn.init.xavier_normal_(tensor,)
        for tensor in [self.attention_b]:
            nn.init.zeros_(tensor,)

        self.drop = nn.Dropout(self.drop_rate)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        embeds_vec_list = inputs
        row, col = [], []

        for r, c in itertools.combinations(embeds_vec_list, 2):
            row.append(r)
            col.append(c)

        p = torch.cat(row, dim=1)
        q = torch.cat(col, dim=1)
        inner_product = p * q

        bi_interaction = inner_product

        #其中nn.ReLU作为一个层结构，必须添加到nn.Module容器中才能使用，
        # 而F.ReLU则作为一个函数调用，看上去作为一个函数调用更方便更简洁。具
        # 体使用哪种方式，取决于编程风格。在PyTorch中,nn.X都有对应的函数版本F.X，
        # 但是并不是所有的F.X均可以用于forward或其它代码段中，因为当网络模型训练完毕时，
        # 在存储model时，在forward中的F.X函数中的参数是无法保存的。
        # 也就是说，在forward中，使用的F.X函数一般均没有状态参数，比如F.ReLU，F.avg_pool2d等，
        # 均没有参数，它们可以用在任何代码片段中。

        attention_temp = self.relu(torch.tensordot(
            bi_interaction, self.attention_W, dims=([-1], [0])) + self.attention_b)
        normalized_att_score = self.softmax(torch.tensordot(
            attention_temp, self.projection_h, dims=([-1], [0])))
        attention_output = torch.sum(normalized_att_score * bi_interaction, dim=1)
        attention_output = self.drop(attention_output)
        afm_out = torch.tensordot(attention_output, self.projection_p, dims=([-1], [0]))

        #print(afm_out)
        return  afm_out

class AFM(nn.Module):
    def __init__(self, feat_size, embedding_size, linear_feature_columns, dnn_feature_columns,
                 use_attention=True, attention_factor=8, l2_reg=0.00001, drop_rate=0.9):
        super(AFM, self).__init__()

        self.sparse_feature_columns = list(filter(lambda x: x[1]=='sparse', dnn_feature_columns))
        self.embedding_dic = nn.ModuleDict({
            feat[0]:nn.Embedding(feat_size[feat[0]], embedding_size, sparse=False) for feat in self.sparse_feature_columns
        })
        self.feature_index = defaultdict(int)
        start = 0
        for feat in feat_size:
            self.feature_index[feat] = start
            start += 1

        self.use_attention = use_attention
        if self.use_attention:
            self.fm = AFMLayer(embedding_size, attention_factor, l2_reg, drop_rate)
        else:
            self.fm = FM()

        dnn_hidden_units = [len(feat_size), 1]
        self.linear = nn.ModuleList([
            nn.Linear(dnn_hidden_units[i], dnn_hidden_units[i + 1]) for i in range(len(dnn_hidden_units) - 1)
        ])
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.00001)
        self.out = nn.Sigmoid()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, X):
        sparse_embedding = [self.embedding_dic[feat[0]](X[:, self.feature_index[feat[0]]].long()).reshape(X.shape[0], 1, -1)
                            for feat in self.sparse_feature_columns]
        logit = X
        for i in range(len(self.linear)):
            fc = self.linear[i](logit)
            fc = self.act(fc)
            fc = self.dropout(fc)
            logit = fc

        if self.use_attention:
            logit += self.fm(sparse_embedding)
        else:
            logit += self.fm(torch.cat(sparse_embedding, dim=1))

        y_pred = torch.sigmoid(logit) # 这里踩了个坑 最开始写成立 nn.softmax(dim=1) 结果训练集 验证集loss都不降

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
    model = AFM(feat_sizes, embedding_size, linear_feature_columns, dnn_feature_columns, use_attention=True).to(device)

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


