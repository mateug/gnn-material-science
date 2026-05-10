import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import CGConv, global_mean_pool

class MixedDiffusionGNN(torch.nn.Module):
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
        super(MixedDiffusionGNN, self).__init__()
        torch.manual_seed(12345)

        # Hiperparámetros de arquitectura
        hidden_dim = 256  # Dimensión aumentada para evitar el cuello de botella de la DGNN
        edge_dim = 1      # Dimensión de los atributos de las aristas (distancias)

        # 1. Proyección inicial: Eleva las características de los nodos a un espacio latente rico.
        # Esto permite que la red capture interacciones complejas desde la primera capa.
        self.node_embedding = Linear(features_channels, hidden_dim)

        # 2. Capas de Convolución de Cristal (CGConv):
        # A diferencia de GINEConv o GraphConv, CGConv implementa un mecanismo de "gate" 
        # que utiliza la información de la arista (distancia) para filtrar la información del nodo.
        # Es ideal para difusión iónica donde la distancia entre sitios es crítica.
        self.conv1 = CGConv(channels=hidden_dim, dim=edge_dim, aggr="mean")
        self.conv2 = CGConv(channels=hidden_dim, dim=edge_dim, aggr="mean")
        self.conv3 = CGConv(channels=hidden_dim, dim=edge_dim, aggr="mean")

        # 3. Bloque de Regresión (MLP):
        # Transforma las representaciones aprendidas en las predicciones de energía.
        self.lin1 = Linear(hidden_dim, 128)
        self.lin2 = Linear(128, 64)
        self.lin3 = Linear(64, 32)
        
        # Capa de salida: predice el vector [Ea_1D, Ea_2D, Ea_3D]
        self.out = Linear(32, n_outputs)

        self.pdropout = pdropout

    def forward(self, batch):
        # Extracción de datos del lote (batch)
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        # Asegurar que edge_attr tenga la dimensión correcta para CGConv
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # 🔷 Embedding inicial
        x = self.node_embedding(x)
        x = F.relu(x)

        # 🔷 Paso de mensajes (Message Passing) con conexiones residuales
        identity = x
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = x + identity # Conexión residual para facilitar el flujo de gradiente

        x = F.relu(self.conv3(x, edge_index, edge_attr))

        # 🔷 Global Pooling
        # El promedio global es más representativo para propiedades de percolación 
        # que el GlobalAttention en modelos de filtrado rápido.
        x = global_mean_pool(x, batch_idx)

        # 🔷 Post-procesamiento y Dropout
        x = F.dropout(x, p=self.pdropout, training=self.training)
        
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))

        # 🔷 Salida final
        return self.out(x)

def load_model(
        n_node_features,
        pdropout=0,
        n_outputs=3,
        **kwargs
):
    """Función de utilidad para instanciar el modelo FDGNN."""
    return MixedDiffusionGNN(
        features_channels=n_node_features,
        pdropout=pdropout,
        n_outputs=n_outputs
    )
