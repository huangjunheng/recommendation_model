import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from config import DeepConnConfig
from data_set import DeepConnDataset
from model import DeepCoNN
from pre_precessing import load_embedding_weights,  get_review_dict
from utils import train_model, val_iter
import torch
from torch.utils.data import DataLoader
path = '../data/office/'
device = torch.device('cuda:0')

df = pd.read_json(path+'reviews.json', lines=True)
train, test = train_test_split(df, test_size=0.2, random_state=3)
train, dev = train_test_split(train, test_size=0.2, random_state=4)


config = DeepConnConfig()
model = DeepCoNN(config, load_embedding_weights())
train_model(model, train, dev, config)

review_by_user, review_by_item = get_review_dict('test')
test_dataset = DeepConnDataset(test, review_by_user, review_by_item, config)
test_dataload = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)

model = torch.load(path+'best_model/best_model').to(device)
mse = val_iter(model, test_dataload)
print('test mse is {}'.format(mse))
