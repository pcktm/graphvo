import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GATv2Conv, SoftmaxAggregation


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.35):
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

        self.conv1 = conv(1, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = conv(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = conv(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = conv(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = conv(512, 512, kernel_size=3, stride=1, padding=1)

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
        return x


class GraphConvolution(nn.Module):
    def __init__(self, dropout=0.5):
        super(GraphConvolution, self).__init__()
        self.conv1 = GraphConv(513, 256, aggr="softmax")
        self.conv2 = GraphConv(257, 64)

        self.dropout = nn.Dropout(dropout)

        self.position = nn.Linear(65, 3)
        self.rotation = nn.Linear(65, 4)  # quaternion

    def forward(self, x, edge_index):
        x = self.dropout(x)
        node_index = torch.arange(x.shape[0]).view(-1, 1).float().to(x.device)
        x = torch.cat((x, node_index), dim=1)
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = torch.cat((x, node_index), dim=1)
        x = F.leaky_relu(self.conv2(x, edge_index))

        x = torch.cat((x, node_index), dim=1)
        position = self.position(x)
        rotation = self.rotation(x)

        rotation = F.normalize(rotation, dim=1)

        return torch.cat((position, rotation), dim=1)


class PoseGNN(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.graph_conv1 = GraphConv(2049, 256, aggr="softmax")
        self.graph_conv2 = GraphConv(257, 128, aggr="softmax")
        self.graph_conv3 = GraphConv(129, 64, aggr="softmax")
        self.dropout = nn.Dropout(dropout)

        self.position = nn.Linear(65, 3)
        self.rotation = nn.Linear(65, 4)  # quaternion

    def forward(self, x, edge_index):
        node_index = torch.arange(x.shape[0]).view(-1, 1).float().to(x.device)
        x = torch.cat((x, node_index), dim=1)
        x = F.leaky_relu(self.graph_conv1(x, edge_index))
        x = self.dropout(x)
        x = torch.cat((x, node_index), dim=1)
        x = F.leaky_relu(self.graph_conv2(x, edge_index))
        x = torch.cat((x, node_index), dim=1)
        x = F.leaky_relu(self.graph_conv3(x, edge_index))

        x = torch.cat((x, node_index), dim=1)
        position = self.position(x)
        rotation = self.rotation(x)

        rotation = F.normalize(rotation, dim=1)

        return torch.cat((position, rotation), dim=1)


class GraphVO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # initially the input is 64x64 images, use standard GraphConv, not my custom module
        self.extract_features = ExtractFeatures()
        self.graph_convolution = GraphConvolution(dropout=0.45)

    def forward(self, x, edge_index):
        # from each image extract features and replace the image with the features
        x = x.view(-1, 1, 64, 64)
        x = self.extract_features(x)
        x = x.view(-1, 512)
        # run graph convolution
        x = self.graph_convolution(x, edge_index)
        return x
