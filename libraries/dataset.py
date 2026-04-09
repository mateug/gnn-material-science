import matplotlib.pyplot as plt
import seaborn           as sns
import numpy             as np
import torch
import json
import os

from pymatgen.io.vasp import Poscar
import libraries.graph as clg

from torch_geometric.data import Data

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sns.set_theme()

def load_structure_from_POSCAR(
        material_folder
):
    """Load a pymatgen Structure from a POSCAR file inside a material folder."""
    poscar_path = os.path.join(material_folder, 'POSCAR')
    if not os.path.exists(poscar_path):
        raise FileNotFoundError(f'POSCAR not found in {material_folder}')
    return Poscar.from_file(poscar_path).structure


def load_material_metadata(
        material_folder
):
    """Load metadata from metadata.json if present."""
    metadata_path = os.path.join(material_folder, 'metadata.json')
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, 'r') as json_file:
        return json.load(json_file)


def get_target_value(
        material_folder,
        target,
        metadata
):
    """Extract a target value from metadata or legacy text files."""
    target_map = {
        'EPA': 'energy_per_atom',
        'bandgap': 'band_gap'
    }
    metadata_key = target_map.get(target, target)
    if metadata_key in metadata:
        return float(metadata[metadata_key])

    if target == 'EPA':
        file_path = os.path.join(material_folder, 'EPA')
    elif target == 'bandgap':
        file_path = os.path.join(material_folder, 'bandgap')
    else:
        raise ValueError(f'Unsupported target {target}')

    if os.path.exists(file_path):
        return float(np.loadtxt(file_path))

    raise FileNotFoundError(f'Target {target} not found for {material_folder}')


def generate_dataset(
        data_path,
        targets,
        data_folder,
        max_samples=None
):
    """Generates a dataset from the raw data and saves it to disk, supporting incremental growth.

    Args:
        data_path   (str): Path to the raw data.
        targets     (list): List of targets to be predicted.
        data_folder (str): Path to the folder where the dataset will be saved.
        max_samples (int, optional): Maximum number of samples to include in the dataset. If None, include all available samples.

    Returns:
        None
    """
    dataset_path = f'{data_folder}/dataset.pt'
    dataset_parameters_path = os.path.join(data_folder, 'dataset_parameters.json')
    
    # Load existing dataset if present
    if os.path.exists(dataset_path):
        dataset = torch.load(dataset_path, weights_only=False)
        processed_labels = {graph.label for graph in dataset}
        print(f"Loaded existing dataset with {len(dataset)} samples.")
    else:
        dataset = []
        processed_labels = set()
        print("Starting new dataset generation.")
    
    # Load or create dataset parameters
    if os.path.exists(dataset_parameters_path):
        with open(dataset_parameters_path, 'r') as f:
            dataset_parameters = json.load(f)
        # Update max_samples if changed
        if dataset_parameters.get('max_samples') != max_samples:
            print(f"Updating max_samples from {dataset_parameters.get('max_samples')} to {max_samples}")
            dataset_parameters['max_samples'] = max_samples
    else:
        dataset_parameters = {
            'input_folder':  data_path,
            'output_folder': data_folder,
            'target':        targets,
            'max_samples':   max_samples
        }
    
    # Check if we need to add more samples
    current_samples = len(dataset)
    if max_samples is not None and current_samples >= max_samples:
        print(f"Dataset already has {current_samples} samples, which meets or exceeds max_samples={max_samples}. Skipping generation.")
        return
    elif max_samples is None and current_samples > 0:
        print(f"Dataset has {current_samples} samples. Since max_samples=None, checking if all materials are processed...")
        # For max_samples=None, we need to check if we've processed all possible materials
        # This is complex, so for now, we'll assume if current_samples > 0 and max_samples=None, we continue to add more
        pass
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
    
    # Save updated dataset parameters
    with open(dataset_parameters_path, 'w') as json_file:
        json.dump(dataset_parameters, json_file)

    # Continue generating the dataset incrementally
    
    # Read all materials within the database
    new_samples_added = 0
    for material in sorted(os.listdir(data_path)):  # Sort for deterministic order
        # Check if max_samples reached
        if max_samples is not None and len(dataset) >= max_samples:
            break
            
        # Check polymorph is a folder
        path_to_material = f'{data_path}/{material}'
        if not os.path.isdir(path_to_material):
            continue
        
        print(material)
        for polymorph in sorted(os.listdir(path_to_material)):  # Sort for deterministic order
            # Check if max_samples reached
            if max_samples is not None and len(dataset) >= max_samples:
                break
                
            label = f'{material} {polymorph}'
            if label in processed_labels:
                continue  # Already processed
                
            # Path to folder containing the POSCAR
            path_to_POSCAR = os.path.join(data_path, material, polymorph)
            
            # Check that the folder is valid and contains a POSCAR
            if not os.path.isdir(path_to_POSCAR):
                continue
            if not os.path.exists(os.path.join(path_to_POSCAR, 'POSCAR')):
                continue

            print(f'\t{polymorph}')
            
            try:
                structure = load_structure_from_POSCAR(path_to_POSCAR)
                nodes, edges, attributes = clg.graph_POSCAR_encoding(structure, encoding_type='sphere-images')
            except Exception as error:
                print(f'\tError: {material} {polymorph} not loaded ({error})')
                continue

            metadata = load_material_metadata(path_to_POSCAR)
            extracted_target = []
            for target in targets:  # Load the properties for the material from metadata or from text files
                try:
                    extracted_target.append(get_target_value(path_to_POSCAR, target, metadata))
                except Exception as error:
                    print(f'\tError loading target {target} for {material} {polymorph}: {error}')
                    extracted_target = None
                    break

            if extracted_target is None:
                continue

            # Construct temporal graph structure
            graph = Data(x=nodes,
                         edge_index=edges.t().contiguous(),
                         edge_attr=attributes.ravel(),
                         y=torch.tensor(extracted_target, dtype=torch.float),
                         label=label
                        )
    
            # Append to dataset
            dataset.append(graph)
            processed_labels.add(label)
            new_samples_added += 1
        
        # Break outer loop if max_samples reached
        if max_samples is not None and len(dataset) >= max_samples:
            break
    
    print(f"Added {new_samples_added} new samples. Total dataset size: {len(dataset)}")
    
    # Save the updated raw dataset
    torch.save(dataset, dataset_path)
    
    if not dataset:
        raise ValueError(f"No valid graphs found in {data_path}. Check data integrity or target files.")
    
    # Update dataset parameters with current sample count
    dataset_parameters['current_samples'] = len(dataset)
    with open(dataset_parameters_path, 'w') as json_file:
        json.dump(dataset_parameters, json_file)


def standardize_dataset(
        dataset,
        transformation=None
):
    """Standardizes a given dataset (both nodes features and edge attributes).
    Typically, a normal distribution is applied, although it be easily modified to apply other distributions.
    Check those graphs with finite attributes and retains labels accordingly.

    Currently: normal distribution.

    Args:
        dataset        (list): List containing graph structures.
        transformation (str):  Type of transformation strategy for edge attributes (None, 'inverse-quadratic').

    Returns:
        Tuple: A tuple containing the normalized dataset and parameters needed to re-scale predicted properties.
            - dataset_std        (list): Normalized dataset.
            - labels_std         (list): Labels from valid graphs.
            - dataset_parameters (dict): Parameters needed to re-scale predicted properties from the dataset.
    """

    # Clone the dataset and labels
    dataset_std = []
    for graph in dataset:
        if check_finite_attributes(graph):
            dataset_std.append(graph.clone())
        else:
            print(f"Graph {graph.label} failed finite attributes check")

    print(f"Graphs after finite check: {len(dataset_std)}")
    if not dataset_std:
        raise ValueError("No graphs passed finite attributes check. Data may contain NaN/inf values.")

    # Number of graphs
    n_graphs = len(dataset_std)
    
    # Number of features per node
    n_features = dataset_std[0].num_node_features
    
    # Number of features per graph
    n_y = dataset_std[0].y.shape[0]
    
    # Check if non-linear standardization
    if transformation == 'inverse-quadratic':
        for data in dataset_std:
            data.edge_attr = 1 / data.edge_attr.pow(2)

    # Compute means
    target_mean = torch.zeros(n_y)
    for target_index in range(n_y):
        target_mean[target_index] = sum([data.y[target_index] for data in dataset_std]) / n_graphs
    
    edge_mean = sum([data.edge_attr.mean() for data in dataset_std]) / n_graphs
    
    # Compute standard deviations
    target_std = torch.zeros(n_y)
    for target_index in range(n_y):
        target_std[target_index] = torch.sqrt(sum([(data.y[target_index] - target_mean[target_index]).pow(2).sum() for data in dataset_std]) / (n_graphs * (n_graphs - 1)))
    
    edge_std = torch.sqrt(sum([(data.edge_attr - edge_mean).pow(2).sum() for data in dataset_std]) / (n_graphs * (n_graphs - 1)))
    
    # In case we want to increase the values of the normalization
    scale = torch.tensor(1e0)

    target_factor = target_std / scale
    edge_factor   = edge_std   / scale

    # Update normalized values into the database
    for data in dataset_std:
        data.y         = (data.y         - target_mean) / target_factor
        data.edge_attr = (data.edge_attr - edge_mean)   / edge_factor

    # Same for the node features
    feat_mean = torch.zeros(n_features)
    feat_std  = torch.zeros(n_features)
    for feat_index in range(n_features):
        # Compute mean
        temp_feat_mean = sum([data.x[:, feat_index].mean() for data in dataset_std]) / n_graphs
        
        # Compute standard deviations
        temp_feat_std = torch.sqrt(sum([(data.x[:, feat_index] - temp_feat_mean).pow(2).sum() for data in dataset_std]) / (n_graphs * (n_graphs - 1)))

        # Update normalized values into the database
        for data in dataset_std:
            data.x[:, feat_index] = (data.x[:, feat_index] - temp_feat_mean) * scale / temp_feat_std
        
        # Append corresponding values for saving
        feat_mean[feat_index] = temp_feat_mean
        feat_std[feat_index]  = temp_feat_std

    # Create and save as a dictionary
    dataset_parameters = {
        'transformation': transformation,
        'target_mean':    np.array(target_mean.cpu().numpy()),
        'feat_mean':      np.array(feat_mean.cpu().numpy()),
        'edge_mean':      edge_mean.cpu().numpy(),
        'target_std':     np.array(target_std.cpu().numpy()),
        'feat_std':       np.array(feat_std.cpu().numpy()),
        'edge_std':       edge_std.cpu().numpy(),
        'scale':          scale.cpu().numpy()
    }
    return dataset_std, dataset_parameters


def standardize_dataset_from_keys(
        dataset,
        standardized_parameters
):
    """Standardize the dataset. Non-linear normalization is also implemented.

    Args:
        dataset                 (list):  List of graphs in PyTorch Geometric's Data format.
        standardized_parameters (dict):  Parameters needed to re-scale predicted properties from the dataset.

    Returns:
        list: Standardized dataset.
    """

    # Read dataset parameters for re-scaling
    edge_mean   = standardized_parameters['edge_mean']
    feat_mean   = standardized_parameters['feat_mean']
    target_mean = standardized_parameters['target_mean']
    scale       = standardized_parameters['scale']
    edge_std    = standardized_parameters['edge_std']
    feat_std    = standardized_parameters['feat_std']
    target_std  = standardized_parameters['target_std']

    # Number of features per node
    n_features = dataset[0].num_node_features
    
    # Number of features per graph
    n_y = dataset[0].y.shape[0]
    
    # Check if non-linear standardization
    if standardized_parameters['transformation'] == 'inverse-quadratic':
        for data in dataset:
            data.edge_attr = 1 / data.edge_attr.pow(2)

    target_factor = target_std / scale
    edge_factor   = edge_std / scale
    feat_factor   = feat_std / scale

    for data in dataset:
        data.edge_attr = (data.edge_attr - edge_mean) / edge_factor

    for target_index in range(n_y):
        for data in dataset:
            data.y[target_index] = (data.y[target_index] - target_mean[target_index]) / target_factor[target_index]

    for feat_index in range(n_features):
        for data in dataset:
            data.x[:, feat_index] = (data.x[:, feat_index] - feat_mean[feat_index]) / feat_factor[feat_index]
    return dataset


def check_finite_attributes(
        data
):
    """
    Checks if all node and edge attributes in the graph are finite (i.e., not NaN, inf, or -inf).

    Args:
        data: A graph object containing node attributes (`data.x`) and edge attributes (`data.edge_attr`).

    Returns:
        bool: 
            - True if all node and edge attributes are finite.
            - False if any node or edge attributes are NaN, inf, or -inf.
    """
    # Check node attributes
    if not torch.all(torch.isfinite(data.x)):
        return False

    # Check edge attributes
    if not torch.all(torch.isfinite(data.edge_attr)):
        return False
    return True


def split_dataset(
        train_ratio,
        val_ratio,
        test_ratio,
        dataset
):
    """Splits the dataset into training, validation, and testing datasets.

    Args:
        train_ratio (float): Ratio of the dataset to be used for training.
        val_ratio   (float): Ratio of the dataset to be used for validation.
        test_ratio  (float): Ratio of the dataset to be used for testing.
        dataset     (list):  List of graphs in PyTorch Geometric's Data format.

    Returns:
        Tuple: A tuple containing the training, validation, and testing datasets.
    """
    # Define the sizes of the train, validation and test sets
    # Corresponds to the size wrt the number of unique materials in the dataset
    train_size = int(train_ratio * len(dataset))
    val_size   = int(val_ratio   * len(dataset))
    test_size  = int(test_ratio  * len(dataset))

    np.random.shuffle(dataset)

    # Random, fast splitting
    train_dataset = dataset[:train_size]
    val_dataset   = dataset[train_size:train_size + val_size]
    test_dataset  = dataset[train_size + val_size:train_size + val_size + test_size]

    print(f'Number of training   graphs: {len(train_dataset)}')
    print(f'Number of validation graphs: {len(val_dataset)}')
    print(f'Number of testing    graphs: {len(test_dataset)}')
    return train_dataset, val_dataset, test_dataset


def load_datasets(
        files_names
):
    """Loads the training, validation, and testing datasets from disk.

    Args:
        files_names (dict): Dictionary containing the file names for the training, validation, and testing datasets.

    Returns:
        Tuple: A tuple containing the training, validation, and testing datasets
    """
    train_dataset = torch.load(files_names['train_dataset_std'], weights_only=False)
    val_dataset   = torch.load(files_names['val_dataset_std'],   weights_only=False)
    test_dataset  = torch.load(files_names['test_dataset_std'],  weights_only=False)

    standardized_parameters = load_json(files_names['std_parameters'])
    return train_dataset, val_dataset, test_dataset, standardized_parameters


def save_datasets(
        train_dataset,
        val_dataset,
        test_dataset,
        files_names
):
    """Saves the training, validation, and testing datasets to disk.

    Args:
        train_dataset (list): List of graphs in PyTorch Geometric's Data format.

    Returns:
        None
    """
    torch.save(train_dataset, files_names['train_dataset_std'])
    torch.save(val_dataset,   files_names['val_dataset_std'])
    torch.save(test_dataset,  files_names['test_dataset_std'])


def save_json(
        file,
        file_name
):
    """Saves a dictionary to a JSON file.

    Args:
        file      (dict): Dictionary containing the data to be saved.
        file_name (str):  Path to the JSON file.

    Returns:
        None
    """
    # Convert torch tensors to numpy arrays
    for key, value in file.items():
        try:
            file[key] = value.tolist()
        except:
            pass

    # Dump the dictionary with numpy arrays to a JSON file
    with open(file_name, 'w') as json_file:
        json.dump(file, json_file)


def load_json(
        file_name
):
    """Loads a JSON file and converts torch tensors to torch tensors.

    Args:
        file_name (str): Path to the JSON file.
        to        (str): Convert torch tensors to torch tensors.

    Returns:
        dict: Dictionary containing the data from the JSON file.
    """
    # Load the data from the JSON file
    with open(file_name, 'r') as json_file:
        file = json.load(json_file)

    for key, value in file.items():
        try:
            file[key] = np.array(value, dtype=np.float32)
        except:
            pass
    return file


def parity_plot(
        train=np.array([np.nan, np.nan]),
        validation=np.array([np.nan, np.nan]),
        test=np.array([np.nan, np.nan]),
        figsize=(3, 3),
        save_to=None
):
    """Plots the computed vs. predicted values for the training, validation, and testing datasets.

    Args:
        train       (list): List containing the computed and predicted values for the training dataset.
        validation  (list): List containing the computed and predicted values for the validation dataset.
        test        (list): List containing the computed and predicted values for the testing dataset.
        figsize    (tuple): Size of the figure.

    Returns:
        None
    """
    x_train, y_train = train
    x_val,   y_val   = validation
    x_test,  y_test  = test

    plt.figure(figsize=figsize)

    if np.any(~np.isnan(train)):
        plt.plot(x_train, y_train, '.', label='Train')
    if np.any(~np.isnan(validation)):
        plt.plot(x_val, y_val, '.', label='Validation')
    if np.any(~np.isnan(test)):
        plt.plot(x_test, y_test, '.', label='Test')

    _min_, _max_ = get_min_max(train.flatten(), validation.flatten(), test.flatten())
    plt.xlabel('Computed')
    plt.ylabel('Predicted ')
    plt.plot([_min_, _max_], [_min_, _max_], '-r')
    plt.legend(loc='best')
    if save_to is not None:
        plt.savefig(save_to, dpi=50, bbox_inches='tight')
    plt.show()


def losses_plot(
        train_losses,
        val_losses,
        to_log=True,
        figsize=(3, 3),
        save_to=None
):
    """Plots the training and validation losses.

    Args:
        train_losses (list): List containing the training losses.
        val_losses   (list): List containing the validation losses.
        to_log       (bool): If True, the losses are plotted in log scale.
        figsize      (tuple): Size of the figure.

    Returns:
        None
    """
    if to_log:
        plt.plot(np.log10(train_losses), label='Train loss')
        plt.plot(np.log10(val_losses) , label='Val  loss')
    else:
        plt.plot(train_losses, label='Train loss')
        plt.plot(val_losses,   label='Val  loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    if save_to is not None:
        plt.savefig(save_to, dpi=50, bbox_inches='tight')
    plt.show()


def get_min_max(*data):
    """Determine the minimum and maximum values in a stack of data.

    Args:
        data: list of torch.tensor

    Returns:
        _min_: float
        _max_: float
    """
    stack = np.concatenate(data)
    _min_ = np.nanmin(stack)
    _max_ = np.nanmax(stack)
    return _min_, _max_
