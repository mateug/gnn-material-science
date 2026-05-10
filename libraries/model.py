import os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from libraries.GCNN import load_model as load_gcnn

# Generic device setting used by training and evaluation utilities.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(
        model_type='GCNN',
        n_node_features=None,
        pdropout=0,
        device='cpu',
        model_name=None,
        mode='train',
        pretrained_name=None,
        n_outputs=1,
):
    """Load a model by type and return it on the requested device."""
    if model_type == 'GCNN':
        model = load_gcnn(
            n_node_features=n_node_features,
            pdropout=pdropout,
            device=device,
            model_name=model_name,
            mode=mode,
            n_outputs=n_outputs,
        )
    elif model_type == 'DGNN':
        from libraries.DGNN import load_model as load_dgnn

        model = load_dgnn(
            n_node_features=n_node_features,
            pdropout=pdropout,
            device=device,
            model_name=model_name,
            mode=mode,
            n_outputs=n_outputs,
        )
    elif model_type == 'FDGNN':
        from libraries.FDGNN import load_model as load_fdgnn

        model = load_fdgnn(
            n_node_features=n_node_features,
            pdropout=pdropout,
            device=device,
            model_name=model_name,
            mode=mode,
            n_outputs=n_outputs,
        )
    elif model_type == 'MDGNN':
        from libraries.MDGNN import load_model as load_mdgnn

        model = load_mdgnn(
            n_node_features=n_node_features,
            pdropout=pdropout,
            device=device,
            model_name=model_name,
            mode=mode,
            n_outputs=n_outputs,
        )
    elif model_type == 'FDGNN2':
        from libraries.FDGNN2 import load_model as load_fdgnn2

        model = load_fdgnn2(
            n_node_features=n_node_features,
            pdropout=pdropout,
            device=device,
            model_name=model_name,
            mode=mode,
            n_outputs=n_outputs,
        )
    elif model_type == 'M3GNet':
        from libraries.M3GNet import load_model as load_m3gnet

        model = load_m3gnet(
            n_node_features=n_node_features,
            pdropout=pdropout,
            device=device,
            model_name=model_name,
            mode=mode,
            pretrained_name=pretrained_name,
        )
    else:
        raise ValueError(f'Unknown model_type: {model_type}')

    return model


def train(
        model,
        criterion,
        train_loader,
        target_factor,
        target_mean,
        optimizer
):
    """Train the model using the provided optimizer and criterion."""
    model.train()
    train_loss = 0.0
    predictions = []
    ground_truths = []
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for data in pbar:
        data = data.to(device)
        out = model(data)           # shape: [batch_size, n_targets]
        n_targets = out.shape[-1]
        # PyG concatenates y as [batch_size * n_targets]; reshape to [batch_size, n_targets]
        y = data.y.view(-1, n_targets)
        loss = criterion(out, y)
        train_loss += loss.item()
        predictions.append(out.detach().cpu().numpy())
        ground_truths.append(y.detach().cpu().numpy())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = train_loss / len(train_loader)
    predictions    = np.concatenate(predictions)    * target_factor + target_mean
    ground_truths  = np.concatenate(ground_truths)  * target_factor + target_mean
    return avg_train_loss, predictions, ground_truths


def test(
        model,
        criterion,
        test_loader,
        target_factor,
        target_mean
):
    """Evaluate the model on the provided dataset loader."""
    model.eval()
    test_loss = 0.0
    predictions = []
    ground_truths = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", leave=False)
        for data in pbar:
            data = data.to(device)
            out = model(data)           # shape: [batch_size, n_targets]
            n_targets = out.shape[-1]
            # PyG concatenates y as [batch_size * n_targets]; reshape to [batch_size, n_targets]
            y = data.y.view(-1, n_targets)
            loss = criterion(out, y)
            test_loss += loss.item()
            predictions.append(out.detach().cpu().numpy())
            ground_truths.append(y.detach().cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    predictions   = np.concatenate(predictions)   * target_factor + target_mean
    ground_truths = np.concatenate(ground_truths) * target_factor + target_mean
    return avg_test_loss, predictions, ground_truths


def forward_predictions(
        reference_dataset,
        pred_dataset,
        model,
        standardized_parameters
):
    """Make predictions on a dataset using a trained model."""
    model.eval()
    target_mean = standardized_parameters['target_mean']
    target_std = standardized_parameters['target_std']
    target_scale = standardized_parameters['scale']

    dataset = DataLoader(pred_dataset, batch_size=128, shuffle=False, pin_memory=True)
    predictions = []
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            pred = model(data).flatten().detach().cpu().numpy()
            predictions.append(pred)

    predictions = np.concatenate(predictions) * target_std / target_scale + target_mean
    return predictions


class EarlyStopping():
    def __init__(
            self,
            patience=5,
            delta=0,
            model_name='model.pt'
    ):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.model_name = model_name

    def __call__(
            self,
            val_loss,
            model
    ):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(
            self,
            val_loss,
            model
    ):
        if val_loss < self.val_loss_min:
            # Si el modelo está envuelto en DataParallel, usamos model.module
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_dict, self.model_name)
            self.val_loss_min = val_loss
