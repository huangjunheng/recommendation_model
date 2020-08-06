import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from config import DeepConnConfig
from sklearn.model_selection import train_test_split
from pre_precessing import get_review_dict


class DeepConnDataset(Dataset):
    def __init__(self, data, user_review_dict, item_review_dict, config):
        super(DeepConnDataset, self).__init__()
        self.data = data
        self.user_review_dict = user_review_dict
        self.item_review_dict = item_review_dict
        self.config = config

        self.user_review, self.user_id, self.item_ids_per_review = self.load_user_review_data()
        self.item_review, self.item_id, self.user_ids_per_review = self.load_item_review_data()
        ratings = self.data['rating'].to_list()
        self.ratings = torch.Tensor(ratings).view(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.user_review[idx], self.item_review[idx], self.ratings[idx]

    def load_user_review_data(self):
        user_reviews = []
        user_ids = []
        item_ids_per_review = []

        for user_id, item_id in zip(self.data['userID'], self.data['itemID']):
            u_review, i_id = self.load_reviews(self.user_review_dict, user_id, item_id)
            user_reviews.append(u_review)
            user_ids.append(user_id)
            item_ids_per_review.append(i_id)

        return torch.LongTensor(user_reviews), torch.LongTensor(user_ids).view(-1, 1), torch.LongTensor(
            item_ids_per_review)

    def load_item_review_data(self):
        item_reviews = []
        item_ids = []
        user_ids_per_review = []

        for user_id, item_id, in zip(self.data['userID'], self.data['itemID']):
            i_review, u_id = self.load_reviews(self.item_review_dict, item_id, user_id)
            item_reviews.append(i_review)
            item_ids.append(item_id)
            user_ids_per_review.append(u_id)

        return torch.LongTensor(item_reviews), torch.LongTensor(item_ids).view(-1, 1), torch.LongTensor(
            user_ids_per_review)

    def load_reviews(self, reviews, query_id, exclude_id):
        # 此函数目的为 如有 数据为 user1 item1， 如果user1评论过item1, 则在训练时先将这条评论masked
        config = self.config
        reviews = reviews[query_id]
        if 'userID' in reviews.columns:
            id_name = 'userID'
            pad_id = config.pad_user_id
        else:
            id_name = 'itemID'
            pad_id = config.pad_item_id

        ids = reviews[id_name][reviews[id_name] != exclude_id].to_list()
        reviews = reviews['review'][reviews[id_name] != exclude_id].to_list()

        # 将每条评论变为定长
        reviews = [r[:config.review_length] for r in reviews]
        # 小于固定长度的 padding
        reviews = [r + [config.pad_word_id] * (config.review_length - len(r)) for r in reviews]

        # 每个用户/物品取固定个数的评论
        reviews = reviews[:config.review_count]
        ids = ids[:config.review_count]
        # 不够的 padding
        pad_length = config.review_count - len(reviews)
        pad_review = [config.pad_word_id] * config.review_length
        reviews += [pad_review] * pad_length
        ids += [pad_id] * pad_length
        return reviews, ids


def main():
    path = '../data/instrument/'

    df = pd.read_json(path + 'reviews.json', lines=True)
    train, test = train_test_split(df, test_size=0.2, random_state=3)
    train, dev = train_test_split(train, test_size=0.2, random_state=4)
    review_by_user, review_by_item = get_review_dict('train')

    config = DeepConnConfig()
    dataset = DeepConnDataset(train, review_by_user, review_by_item, config)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    for user_review, item_review, ratings in loader:
        print(user_review.shape)


if __name__ == '__main__':
    main()
