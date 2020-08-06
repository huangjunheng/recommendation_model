import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from pre_precessing import get_review_dict
from data_set import DeepConnDataset
from  sklearn.metrics import mean_squared_error
import time

device = torch.device('cuda:0')
path = '../data/office/'


def val_iter(model, data_load):
    model.eval()
    labels, predicts = list(), list()
    with torch.no_grad():
        for user_review, item_review, ratings in data_load:
            user_review, item_review, ratings = user_review.to(device), item_review.to(device), ratings.to(device)
            y_pre = model(user_review, item_review)
            labels.extend(ratings.tolist())
            predicts.extend(y_pre.tolist())
    mse = mean_squared_error(np.array(labels), np.array(predicts))
    return mse

def train_model(model, train_data, val_data, config):

    start_time = time.time()

    config = config
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_s = lr_scheduler.ExponentialLR(opt, gamma=config.learning_rate_decay)
    loss_func = nn.MSELoss()

    review_by_user, review_by_item = get_review_dict('train')
    train_dataset = DeepConnDataset(train_data, review_by_user, review_by_item, config)
    train_dataload = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=False)
    val_dataset = DeepConnDataset(val_data, review_by_user, review_by_item, config)
    val_dataload = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=False)

    cur_epoch = 0
    best_mse, best_epoch = 10, 0
    while cur_epoch < config.num_epochs:
        model.train()
        total_loss, total_len = 0, 0
        for user_review, item_review, ratings in train_dataload:
            user_review, item_review, ratings = user_review.to(device), item_review.to(device), ratings.to(device)
            predict = model(user_review, item_review)

            loss = loss_func(predict, ratings)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(predict)
            total_len += len(predict)
        loss = total_loss / total_len

        mse = val_iter(model, val_dataload)
        print('epoch {}, train_loss is {}, val mse is {}'.format(cur_epoch, loss, mse))

        if mse < best_mse:
            best_mse, best_epoch = mse, cur_epoch
            torch.save(model, path+'best_model/best_model')
        cur_epoch += 1

    end_time = time.time()
    print('train used time {} min'.format((end_time-start_time)/60.0))





