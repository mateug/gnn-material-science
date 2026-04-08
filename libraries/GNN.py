import numpy               as np
import matplotlib.pyplot   as plt
import matplotlib.patches  as patches
import matplotlib.cm       as cm
import torch.nn.functional as F
import pandas              as pd
import sys
import re
import torch
import yaml
import collections
import os

from matplotlib.colors  import Normalize
from torch.nn           import Linear
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNN(
    torch.nn.Module
):
    """Graph convolution neural network.
    """
    
    def __init__(
            self,
            features_channels,
            pdropout
    ):
        """Initializes the Graph Convolutional Neural Network.

        Args:
            features_channels (int):   Number of input features.
            pdropout          (float): Dropout probability for regularization.

        Returns:
            None
        """
        
        super(GCNN, self).__init__()
        
        # Set random seed for reproducibility
        torch.manual_seed(12345)
        
        # Define graph convolution layers
        self.conv1 = GraphConv(features_channels, 512)
        self.conv2 = GraphConv(512, 512)
        self.conv3 = GraphConv(512, 256)
        
        # Define linear layers
        self.linconv1 = Linear(256, 64)
        self.linconv2 = Linear(64, 1)

        self.norm1 = torch.nn.BatchNorm1d(512)
        
        self.pdropout = pdropout

    def forward(
            self,
            x,
            edge_index,
            edge_attr,
            batch
    ):
        ## CONVOLUTION
        
        # Apply graph convolution with ReLU activation function
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        x = x.relu()

        ## POOLING
        
        # Apply global mean pooling to reduce dimensionality
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # Apply dropout regularization
        x = F.dropout(x, p=self.pdropout, training=self.training)
        
        # Apply linear convolution with ReLU activation function
        x = self.linconv1(x)
        x = x.relu()
        x = self.linconv2(x)
        x = x.relu()
        return x


def train(
        model,
        criterion,
        train_loader,
        target_factor,
        target_mean,
        optimizer
):
    """Train the model using the provided optimizer and criterion on the training dataset.

    Args:
        model        (torch.nn.Module):             The model to train.
        optimizer    (torch.optim.Optimizer):       The optimizer to use for updating model parameters.
        criterion    (torch.nn.Module):             The loss function to use.
        train_loader (torch.utils.data.DataLoader): The training dataset loader.

    Returns:
        float: The average training loss.
    """
    
    model.train()
    train_loss = 0
    all_predictions   = []
    all_ground_truths = []
    for data in train_loader:  # Iterate in batches over the training dataset
        # Moving data to device
        data = data.to(device)
        
        # Perform a single forward pass
        out = model(data.x, data.edge_index, data.edge_attr, data.batch).to(device).flatten()
        
        # Compute the loss
        loss = criterion(out, data.y)
        
        # Accumulate the training loss
        train_loss += loss.item()

        # Append predictions and ground truths to lists
        all_predictions.append(out.detach())
        all_ground_truths.append(data.y.detach())
        
        # Derive gradients
        loss.backward()
        
        # Update parameters based on gradients
        optimizer.step()
        
        # Clear gradients
        optimizer.zero_grad()
    
    # Compute the average training loss
    avg_train_loss = train_loss / len(train_loader)
    
    # Concatenate predictions and ground truths into single arrays
    all_predictions = torch.cat(all_predictions) * target_factor + target_mean
    all_ground_truths = torch.cat(all_ground_truths) * target_factor + target_mean
    
    return avg_train_loss, all_predictions.cpu().numpy(), all_ground_truths.cpu().numpy()


def test(
        model,
        criterion,
        test_loader,
        target_factor,
        target_mean
):
    """Evaluate the performance of a given model on a test dataset.

    Args:
        model       (torch.nn.Module):             The model to evaluate.
        criterion   (torch.nn.Module):             The loss function to use.
        test_loader (torch.utils.data.DataLoader): The test dataset loader.

    Returns:
        float: The average loss on the test dataset.
    """
    
    model.eval()
    test_loss = 0
    all_predictions   = []
    all_ground_truths = []
    with torch.no_grad():
        for data in test_loader:  # Iteratea in batches over the train/test dataset
            # Moving data to device
            data = data.to(device)
            
            # Perform a single forward pass
            out = model(data.x, data.edge_index, data.edge_attr, data.batch).to(device).flatten()
            
            # Compute the loss
            loss = criterion(out, data.y)
            
            # Accumulate the training loss
            test_loss += loss.item()

            # Append predictions and ground truths to lists
            all_predictions.append(out.detach())
            all_ground_truths.append(data.y.detach())
    
    # Compute the average test loss
    avg_test_loss = test_loss / len(test_loader)
    
    # Concatenate predictions and ground truths into single arrays
    all_predictions = torch.cat(all_predictions) * target_factor + target_mean
    all_ground_truths = torch.cat(all_ground_truths) * target_factor + target_mean
    
    return avg_test_loss, all_predictions.cpu().numpy(), all_ground_truths.cpu().numpy()


