import torch
import torch.nn as nn
import torch.nn.functional as F
from fastkan import FastKAN as KAN


class CustomCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(CustomCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.conv_unigram = nn.Conv2d(1, 256, (1, embed_dim), padding=(0, 0))
        self.bn_unigram = nn.BatchNorm2d(256)
        self.pool_unigram = nn.MaxPool1d(2)
        self.dropout_unigram = nn.Dropout(0.25)

        self.conv_trigram = nn.Conv2d(1, 256, (3, embed_dim), padding=(1, 0))
        self.bn_trigram = nn.BatchNorm2d(256)
        self.pool_trigram = nn.MaxPool1d(2)
        self.dropout_trigram = nn.Dropout(0.25)

        self.conv_pentagram = nn.Conv2d(1, 256, (5, embed_dim), padding=(2, 0))
        self.bn_pentagram = nn.BatchNorm2d(256)
        self.pool_pentagram = nn.MaxPool1d(2)
        self.dropout_pentagram = nn.Dropout(0.25)

        self.conv_pcat1 = nn.Conv2d(3 * 256, 1024, (3, 3), padding=(1, 1))
        self.bn_pcat1 = nn.BatchNorm2d(1024)
        self.pool_pcat1 = nn.MaxPool2d((2, 1))
        self.dropout_pcat1 = nn.Dropout(0.25)

        self.conv_pcat2 = nn.Conv2d(1024, 2048, (3, 3), padding=(1, 1))
        self.bn_pcat2 = nn.BatchNorm2d(2048)
        self.pool_pcat2 = nn.MaxPool2d((2, 1))
        self.dropout_pcat2 = nn.Dropout(0.25)

        self.conv_pcat3 = nn.Conv2d(2048, 4096, (3, 3), padding=(1, 1))
        self.bn_pcat3 = nn.BatchNorm2d(4096)
        self.pool_pcat3 = nn.MaxPool2d((4, 1))
        self.dropout_pcat3 = nn.Dropout(0.25)

        self.kan_pre = KAN([32768, 256], num_grids=8)
        self.kan = KAN([256, 64, 8, 2], num_grids=8)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)

        x1 = self.dropout_unigram(self.pool_unigram(F.relu(self.bn_unigram(self.conv_unigram(x))).squeeze(3)))
        x2 = self.dropout_trigram(self.pool_trigram(F.relu(self.bn_trigram(self.conv_trigram(x))).squeeze(3)))
        x3 = self.dropout_pentagram(self.pool_pentagram(F.relu(self.bn_pentagram(self.conv_pentagram(x))).squeeze(3)))

        x_cat = torch.cat((x1, x2, x3), dim=1).unsqueeze(3)

        x_out = self.dropout_pcat1(self.pool_pcat1(F.relu(self.bn_pcat1(self.conv_pcat1(x_cat)))))
        x_out = self.dropout_pcat2(self.pool_pcat2(F.relu(self.bn_pcat2(self.conv_pcat2(x_out)))))
        x_out = self.dropout_pcat3(self.pool_pcat3(F.relu(self.bn_pcat3(self.conv_pcat3(x_out)))))

        x_out = x_out.view(x_out.size(0), -1)

        x_out = self.kan_pre(x_out)
        x_out = self.kan(x_out)

        return x_out
