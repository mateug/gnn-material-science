import os
import torch.nn.functional as F
import torch.nn as nn
import torch

from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import NNConv, GlobalAttention


class DiffusionGNN(torch.nn.Module):
    """Graph neural network tailored for ionic diffusion (Ea prediction)."""

    def __init__(
            self,
            features_channels,
            pdropout
    ):
        super(DiffusionGNN, self).__init__()
        torch.manual_seed(12345)

        hidden = 128

        # 🔷 Edge networks (usan distancia como input)
        self.edge_mlp1 = Sequential(
            Linear(1, 64),
            ReLU(),
            Linear(64, features_channels * hidden)
        )

        self.edge_mlp2 = Sequential(
            Linear(1, 64),
            ReLU(),
            Linear(64, hidden * hidden)
        )

        self.edge_mlp3 = Sequential(
            Linear(1, 64),
            ReLU(),
            Linear(64, hidden * hidden)
        )

        # 🔷 Convoluciones tipo MPNN (dependientes de edge_attr)
        self.conv1 = NNConv(features_channels, hidden, self.edge_mlp1, aggr='mean')
        self.conv2 = NNConv(hidden, hidden, self.edge_mlp2, aggr='mean')
        self.conv3 = NNConv(hidden, hidden, self.edge_mlp3, aggr='mean')

        # 🔷 Atención global (mejor que mean pooling)
        self.att_pool = GlobalAttention(
            gate_nn=Sequential(
                Linear(hidden, 64),
                ReLU(),
                Linear(64, 1)
            )
        )

        # 🔷 MLP final
        self.lin1 = Linear(hidden, 128)
        self.lin2 = Linear(128, 64)
        self.lin3 = Linear(64, 16)
        self.lin = Linear(16, 1)

        self.pdropout = pdropout

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr.unsqueeze(-1)  # 🔴 MUY IMPORTANTE
        batch_index = batch.batch

        # 🔷 Message passing con residuals
        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = x1.relu()

        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = (x2 + x1).relu()

        x3 = self.conv3(x2, edge_index, edge_attr)
        x3 = (x3 + x2).relu()

        # 🔷 Pooling
        x = self.att_pool(x3, batch_index)

        x = F.dropout(x, p=self.pdropout, training=self.training)

        # 🔷 MLP final
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        x = self.lin3(x).relu()
        x = self.lin(x)

        return x


def load_model(
        n_node_features,
        pdropout=0,
        device='cpu',
        model_name=None,
        mode='eval'
):
    model = DiffusionGNN(features_channels=n_node_features, pdropout=pdropout)

    if model_name is not None and os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location='cpu'))

    model = model.to(device)

    if mode == 'eval':
        model.eval()
    elif mode == 'train':
        model.train()

    model = nn.DataParallel(model)
    return model