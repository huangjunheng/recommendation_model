class DeepConnConfig():
    num_epochs = 20
    batch_size = 16
    learning_rate = 1e-3
    l2_regularization = 1e-6
    learning_rate_decay = 0.99

    pad_word_id = 3000000
    pad_user_id = 4905 # max user id + 1    1429
    pad_item_id = 2420 # max item id +1     900

    review_length = 125 # max review length -5
    review_count = 40 # max review count - 5

    word_dim = 300
    id_dim = 32

    kernel_width = 5
    kernel_deep = 100

    dropout = 0.5