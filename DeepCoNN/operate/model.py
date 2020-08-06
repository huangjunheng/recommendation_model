import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizationMachine(nn.Module):

    def __init__(self, p, k):
        super(FactorizationMachine, self).__init__()

        self.p, self.k = p, k
        self.linear = nn.Linear(self.p, 1, bias=True)
        self.v = nn.Parameter(torch.zeros(self.p, self.k))
        self.drop = nn.Dropout(0.2)

    def fm_layer(self, x):
        linear_part = self.linear(x)
        inter_part1 = torch.mm(x, self.v)
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        pair_interactions = torch.sum(torch.sub(torch.pow(inter_part1, 2),
                                                inter_part2), dim=1)
        self.drop(pair_interactions)
        output = linear_part.transpose(1, 0) + 0.5 * pair_interactions
        return output

    def forward(self, x):
        output = self.fm_layer(x)
        return output.view(-1, 1)

class DeepCoNN(nn.Module):

    def __init__(self, config, embedding_weight):
        super(DeepCoNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.embedding.weight.requires_grad = False

        self.conv_u = nn.Sequential(
            nn.Conv1d(
                in_channels=config.word_dim,
                out_channels=config.kernel_deep,
                kernel_size=config.kernel_width,
                padding=(config.kernel_width-1)//2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, config.review_length)),
            nn.Dropout(p=config.dropout))
        self.linear_u = nn.Sequential(
            nn.Linear(config.kernel_deep*config.review_count, config.id_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout))

        self.conv_i = nn.Sequential(
            nn.Conv1d(
                in_channels=config.word_dim,
                out_channels=config.kernel_deep,
                kernel_size=config.kernel_width,
                padding=(config.kernel_width - 1) // 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, config.review_length)),
            nn.Dropout(p=config.dropout))
        self.linear_i = nn.Sequential(
            nn.Linear(config.kernel_deep * config.review_count, config.id_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout))

        self.out = FactorizationMachine(config.id_dim * 2, 10)

    def forward(self, user_review, item_review):
        batch_size = user_review.shape[0]
        new_batch_size = user_review.shape[0] * user_review.shape[1]

        user_review = user_review.reshape(new_batch_size, -1)
        item_review = item_review.reshape(new_batch_size, -1)
        u_vec = self.embedding(user_review).permute(0, 2, 1)
        i_vec = self.embedding(item_review).permute(0, 2, 1)

        user_latent = self.conv_u(u_vec).reshape(batch_size, -1)
        item_latent = self.conv_i(i_vec).reshape(batch_size, -1)

        user_latent = self.linear_u(user_latent)
        item_latent = self.linear_i(item_latent)

        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        prediction = self.out(concat_latent)
        return prediction

