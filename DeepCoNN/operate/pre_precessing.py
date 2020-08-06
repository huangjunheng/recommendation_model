import numpy as np
import pandas as pd
import torch
import nltk
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import pickle
from gensim.models.keyedvectors import Word2VecKeyedVectors

path = '../data/office/'
PAD_WORD = '<pad>'
PAD_WORD_ID = 3000000
WORD_EMBEDDINF_SIZE = 300


def process_raw_data(in_path, out_path):
    df = pd.read_json(in_path, lines=True)
    df = df[['reviewerID', 'asin', 'reviewText', 'overall']]
    df.columns = ['userID', 'itemID', 'review', 'rating']

    # 将用户/物品id映射为数字
    df['userID'] = df.groupby(df['userID']).ngroup()
    df['itemID'] = df.groupby(df['itemID']).ngroup()

    with open('../data/embedding_data/stopwords.txt') as f:
        stop_words = set(f.read().splitlines())

    with open('../data/embedding_data/punctuations.txt') as f:
        punctuations = set(f.read().splitlines())

    def clean_review(review):
        lemmatizer = nltk.WordNetLemmatizer()
        review = review.lower()
        for p in punctuations:
            review = review.replace(p, ' ')
        tokens = review.split()
        tokens = [word for word in tokens if word not in stop_words]
        # 词形归并 词干提取
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    df['review'] = df['review'].apply(clean_review)
    df.to_json(out_path, orient='records', lines=True)


def get_word_vec():
    # 加载预训练词嵌入模型
    in_path = '../data/embedding_data/GoogleNews-vectors-negative300.bin'
    out_path = path + 'embedding_weight.pt'
    word_vec = KeyedVectors.load_word2vec_format(in_path, binary=True)
    word_vec.add([PAD_WORD], np.zeros([1, 300]))

    # 保存预训练模型为tensor格式， 以便于后续训练
    weight = torch.Tensor(word_vec.vectors)
    torch.save(weight, out_path)
    return word_vec


def load_embedding_weights(in_path=path + 'embedding_weight.pt'):
    return torch.load(in_path)


def get_reviews_in_idx(data, word_vec):
    def review2wid(review):
        wids = []
        for word in review.split():
            if word in word_vec:
                wid = word_vec.vocab[word].index
            else:
                wid = word_vec.vocab[PAD_WORD].index
            wids.append(wid)
        return wids

    data['review'] = data['review'].apply(review2wid)
    review_by_user = dict(list(data[['itemID', 'review']].groupby(data['userID'])))
    review_by_item = dict(list(data[['userID', 'review']].groupby(data['itemID'])))
    return review_by_user, review_by_item


def get_max_review_length(data, percentile=0.85):
    review_lengths = data['review'].apply(lambda review: len(review.split()))
    max_length = int(review_lengths.quantile(percentile, interpolation='lower'))
    return max_length


def get_max_review_count(data, percentile=0.85):
    review_count_user = data['review'].groupby(data['userID']).count()
    review_count_user = int(review_count_user.quantile(percentile, interpolation='lower'))

    review_count_item = data['review'].groupby(data['itemID']).count()
    review_count_item = int(review_count_item.quantile(percentile, interpolation='lower'))

    return max(review_count_user, review_count_item)


def get_max_user_id(data):
    return max(data['userID'])


def get_max_item_id(data):
    return max(data['itemID'])


def save_review_dict(data, word_vec, data_type):
    user_review, item_review = get_reviews_in_idx(data, word_vec)
    pickle.dump(user_review, open(path + 'user_review_word_idx_{}.p'.format(data_type), 'wb'))
    pickle.dump(item_review, open(path + 'item_review_word_idx_{}.p'.format(data_type), 'wb'))


def get_review_dict(data_type):
    user_review = pickle.load(open(path + 'user_review_word_idx_{}.p'.format(data_type), 'rb'))
    item_review = pickle.load(open(path + 'item_review_word_idx_{}.p'.format(data_type), 'rb'))
    return user_review, item_review


def main():
    process_raw_data(path + 'Office_Products_5.json', path + 'reviews.json')
    df = pd.read_json(path + 'reviews.json', lines=True)
    train, test = train_test_split(df, test_size=0.2, random_state=3)
    train, dev = train_test_split(train, test_size=0.2, random_state=4)
    known_data = pd.concat([train, dev], axis=0)
    all_data = pd.concat([train, dev, test], axis=0)

    print('max review length is {}'.format(get_max_review_length(all_data)))
    print('max review count is {}'.format(get_max_review_count(all_data)))
    print('max user id is {}'.format(get_max_user_id(all_data)))
    print('max item id is {}'.format(get_max_item_id(all_data)))

    word_vec = get_word_vec()

    save_review_dict(known_data, word_vec, 'train')
    save_review_dict(all_data, word_vec, 'test')


if __name__ == '__main__':
    main()
