import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GATv2Conv


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout),
    )


class ExtractFeatures(nn.Module):
    def __init__(self, use_pooling=True):
        super().__init__()
        self.use_pooling = use_pooling
        if self.use_pooling:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = conv(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = conv(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = conv(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = conv(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = conv(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = conv(512, 1024, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_pooling:
            x = self.pool(x)
        x = self.conv2(x)
        if self.use_pooling:
            x = self.pool(x)
        x = self.conv3(x)
        x = self.conv3_1(x)
        if self.use_pooling:
            x = self.pool(x)
        x = self.conv4(x)
        x = self.conv4_1(x)
        if self.use_pooling:
            x = self.pool(x)
        x = self.conv5(x)
        x = self.conv5_1(x)
        if self.use_pooling:
            x = self.pool(x)
        x = self.conv6(x)
        return x


class GraphConvolution(nn.Module):
    def __init__(self, dropout=0):
        super(GraphConvolution, self).__init__()
        self.conv1 = GraphConv(1024, 512)
        self.conv2 = GraphConv(512, 128)
        self.conv3 = GraphConv(128, 64)

        self.dropout = nn.Dropout(dropout)

        self.position = nn.Linear(64, 3)
        self.rotation = nn.Linear(64, 4)

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.leaky_relu(self.conv3(x, edge_index))

        position = self.position(x)
        rotation = self.rotation(x)

        rotation = F.normalize(rotation, p=2, dim=1)

        return torch.cat((position, rotation), dim=1)


class GraphVO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.extract_features = ExtractFeatures()
        self.graph_convolution = GraphConvolution()

    def forward(self, x, edge_index):
        # from each image extract features and replace the image with the features
        # x is of shape (batch_size * graph_length, 3, 64, 64)
        x = x.view(-1, 3, 64, 64)
        x = self.extract_features(x)
        x = x.view(-1, 1024)

        # run graph convolution
        x = self.graph_convolution(x, edge_index)
        return x
    
    