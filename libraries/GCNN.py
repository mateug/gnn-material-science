import os
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn import Linear
from torch_geometric.nn import GraphConv, global_mean_pool


class GCNN(torch.nn.Module):
    """Graph convolution neural network."""

    def __init__(
            self,
            features_channels,
            pdropout,
            n_outputs=1
    ):
        """Initializes the Graph Convolutional Neural Network.

        Args:
            features_channels (int):   Number of input features.
            pdropout          (float): Dropout probability for regularization.
            n_outputs         (int):   Number of output targets (1 for single, 3 for all-energies).
        """
        super(GCNN, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = GraphConv(features_channels, 512)
        self.conv2 = GraphConv(512, 512)

        self.linconv1 = Linear(512, 64)
        self.linconv2 = Linear(64, 16)
        self.lin = Linear(16, n_outputs)
        self.pdropout = pdropout

    def forward(self, batch):
        x = self.conv1(batch.x, batch.edge_index, batch.edge_attr)
        x = x.relu()
        x = self.conv2(x, batch.edge_index, batch.edge_attr)
        x = x.relu()

        x = global_mean_pool(x, batch.batch)
        x = F.dropout(x, p=self.pdropout, training=self.training)

        x = self.linconv1(x)
        x = x.relu()
        x = self.linconv2(x)
        x = x.relu()

        x = self.lin(x)
        return x


def load_model(
        n_node_features,
        pdropout=0,
        device='cpu',
        model_name=None,
        mode='eval',
        n_outputs=1
):
    model = GCNN(features_channels=n_node_features, pdropout=pdropout, n_outputs=n_outputs)

    if model_name is not None and os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location='cpu'))

    model = model.to(device)

    if mode == 'eval':
        model.eval()
    elif mode == 'train':
        model.train()

    model = nn.DataParallel(model)
    return model
