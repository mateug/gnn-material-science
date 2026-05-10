import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import CGConv, global_mean_pool

class FastDiffusionGNN(torch.nn.Module):
    """
    Fast Diffusion Graph Neural Network (FDGNN) diseñada para el filtrado rápido (Fase 2).
    
    Esta arquitectura utiliza Crystal Graph Convolutions (CGConv) para modelar la
    interdependencia entre la química de los nodos y la geometría de las aristas,
    optimizada para predecir energías de activación (Ea) en 1D, 2D y 3D.
    """

    def __init__(
            self,
            features_channels,
            pdropout,
            n_outputs=3
    ):
        """
        Args:
            features_channels (int): Número de características de entrada por nodo.
            pdropout (float): Probabilidad de dropout para regularización.
            n_outputs (int): Número de salidas (por defecto 3: Ea_1D, Ea_2D, Ea_3D).
        """
        super(FastDiffusionGNN, self).__init__()
        torch.manual_seed(12345)

        # Hiperparámetros de arquitectura
        hidden_dim = 192  # Punto medio para balancear velocidad y capacidad
        edge_dim = 1      

        # 1. Proyección inicial
        self.node_embedding = Linear(features_channels, hidden_dim)
        self.bn_init = nn.BatchNorm1d(hidden_dim)

        # 2. Capas de Convolución de Cristal (CGConv) con BatchNorm
        self.conv1 = CGConv(channels=hidden_dim, dim=edge_dim, aggr="mean")
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = CGConv(channels=hidden_dim, dim=edge_dim, aggr="mean")
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 3. Bloque de Regresión (MLP) robusto
        self.lin1 = Linear(hidden_dim, 128)
        self.lin2 = Linear(128, 64)
        
        # Capa de salida
        self.out = Linear(64, n_outputs)

        self.pdropout = pdropout

    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # 🔷 Embedding inicial
        x = self.node_embedding(x)
        x = self.bn_init(x)
        x = F.relu(x)

        # 🔷 Paso de mensajes (Message Passing)
        identity = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = x + identity # Conexión residual

        # 🔷 Global Pooling
        x = global_mean_pool(x, batch_idx)

        # 🔷 Post-procesamiento y Dropout
        x = F.dropout(x, p=self.pdropout, training=self.training)
        
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        # 🔷 Salida final
        return self.out(x)

def load_model(
        n_node_features,
        pdropout=0,
        device='cpu',
        model_name=None,
        mode='eval',
        n_outputs=3,
        **kwargs
):
    """Función de utilidad para instanciar y cargar el modelo FDGNN."""
    model = FastDiffusionGNN(
        features_channels=n_node_features,
        pdropout=pdropout,
        n_outputs=n_outputs
    )

    if model_name is not None and os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location='cpu'))

    model = model.to(device)

    if mode == 'eval':
        model.eval()
    elif mode == 'train':
        model.train()

    model = nn.DataParallel(model)
    return model
