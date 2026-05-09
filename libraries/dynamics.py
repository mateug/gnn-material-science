import numpy as np
import sys
import os
import json

from pymatgen.core                        import Structure, Lattice
from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
from pymatgen.io.vasp                     import Poscar
from sklearn.cluster                      import KMeans, SpectralClustering
from sklearn.metrics                      import silhouette_score
from scipy.spatial.distance               import pdist
from ase.io                               import read

kmeans_kwargs = dict(random_state=0, init='random', n_init=10, max_iter=300, tol=1e-04)
spectral_kwargs = dict(random_state=0, affinity='nearest_neighbors', n_neighbors=1000, assign_labels='cluster_qr')


def compute_rdf(
        vacancy_positions,
        ref_structure_volume,
        r_max=5,
        dr=0.05
):
    """
    r_max  = 5  # maximum distance in Å
    dr     = 0.05  # bin width in Å
    """

    r_bins = np.arange(0, r_max + dr, dr)

    # Compute RDF if there are vacancies
    rdf = np.zeros(len(r_bins) - 1)
    if len(vacancy_positions) > 1:
        vac_coords = np.array(vacancy_positions)
        dist_matrix = pdist(vac_coords)
        rdf_hist, _ = np.histogram(dist_matrix, bins=r_bins)

        # Get the volume of each sphere at distance r
        shell_volumes = (4 / 3) * np.pi * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)

        # Total number of vacancy pairs and number density
        n_vac = len(vacancy_positions)
        rho = n_vac / ref_structure_volume

        # Normalize the distribution based on volume
        rdf = rdf_hist / (rho * shell_volumes * n_vac)
    return rdf


def get_cartesian_coordinates(
        frac_coordinates,
        cell
):
    """
    Returns the cartesian coordinates of a simulation from these in direct form, and the number of configurations.

    Args:
       coordinates (array): The coordinates in direct form.
       cell        (array): The cell parameters.

    Returns:
       array: The cartesian coordinates.
    """
    # Shape the configurations data into the positions attribute
    n_conf = frac_coordinates.shape[0]

    # Get the variation in positions applying periodic boundary condition
    dpos = np.diff(frac_coordinates, axis=0)
    while np.any(dpos > 0.5):
        dpos[dpos > 0.5] -= 1
    while np.any(dpos < -0.5):
        dpos[dpos < -0.5] += 1

    # Copy coordinates
    cart_coordinates = frac_coordinates.copy()

    # Get the positions and variations in cell units
    cart_coordinates[0] = cart_coordinates[0] @ cell
    for i in range(n_conf-1):
        dpos[i] = dpos[i] @ cell

    cart_coordinates = np.concatenate([np.expand_dims(cart_coordinates[0], 0), dpos], axis=0)
    cart_coordinates = np.cumsum(cart_coordinates, axis=0)
    return cart_coordinates


def mean_frac_coordinates(
        frac_coordinates
):
    """
    Compute the mean fractional position of an atom
    accounting for periodic boundary conditions.

    Args:
        frac_coordinates ((N, 3) ndarray): Fractional coordinates in [0,1)

    Returns:
        mean_frac ((3,) ndarray): Mean fractional position in [0,1)
    """
    frac_coordinates = np.asarray(frac_coordinates)

    # Get the variation in positions applying periodic boundary condition
    dpos = np.diff(frac_coordinates, axis=0)
    while np.any(dpos > 0.5):
        dpos[dpos > 0.5] -= 1
    while np.any(dpos < -0.5):
        dpos[dpos < -0.5] += 1

    # Reconstruct unwrapped trajectory
    unwrapped = np.zeros_like(frac_coordinates)
    unwrapped[0] = frac_coordinates[0]
    unwrapped[1:] = frac_coordinates[0] + np.cumsum(dpos, axis=0)

    # Compute mean in unwrapped space
    mean_unwrapped = np.mean(unwrapped, axis=0)

    # Wrap back into [0,1)
    mean_frac = mean_unwrapped % 1.0
    return mean_frac


def select_classifier(
        method,
        n_clusters
):
    """
    Returns the selected classifier, if available, for a selected number of clusters.

    Args:
        method     (str): The clustering method, 'K-means' or 'Spectral'.
        n_clusters (int): The number of clusters.

    Returns:
        sklearn.cluster object: A clustering model based on the specified method and clusters.
    """
    if method == 'K-means':
        return KMeans(n_clusters=n_clusters, **kmeans_kwargs)
    elif method == 'Spectral':
        return SpectralClustering(n_clusters=n_clusters, **spectral_kwargs)
    else:
        sys.exit(f'Clustering method {method} not available')


def calculate_silhouette(
        method,
        coordinates,
        n_attemps=50,
        silhouette_threshold=0.5
):
    """
    The number of clusters is selected as the maximum in the Silhouette coefficient.
    One only cluster is selected is the threshold is not achieved.
    Calculate Silhouette Score for Clustering

    Args:
        method      (str):           The clustering method, 'K-means' or 'Spectral'.
        coordinates (numpy.ndarray): The input coordinates for clustering.
        n_attempts  (int, optional): The number of attempts to evaluate Silhouette scores (default is 10).
        silhouette_threshold (float): Minimum Silhouette score to accept multiple clusters.

    Returns:
        int: The number of clusters determined based on the Silhouette coefficient.
    """
    # Define clusters search space
    all_clusters = np.arange(2, n_attemps + 1)

    # Iterate over each number of clusters
    silhouette_averages = []
    for n_clusters in all_clusters:
        clustering = select_classifier(method, n_clusters)

        # Get labels
        labels = clustering.fit_predict(coordinates)

        # Calculate ratios for a range of clusters
        silhouette_averages.append(silhouette_score(coordinates, labels))

    n_clusters = all_clusters[np.argmax(silhouette_averages)]
    if np.max(silhouette_averages) < silhouette_threshold:
        n_clusters = 1

    return n_clusters

def smooth_classification(
        classification,
        min_length=5
):
    """
    Smooth the classification array by removing short excursions.

    Args:
        classification (list or ndarray): Array of cluster labels.
        min_length     (int): Minimum length of a cluster to be retained.

    Returns:
        ndarray: Smoothed classification array.
    """
    smoothed = classification.copy()
    i = 1
    while i < len(smoothed) - 1:
        curr = smoothed[i]
        prev = smoothed[i - 1]

        # Look for start of a new cluster
        if curr != prev:
            start = i
            while i < len(smoothed) and smoothed[i] == curr:
                i += 1
            end = i

            # Check if it is a short excursion
            if end - start < min_length and start > 0 and end < len(smoothed):
                if smoothed[start - 1] == smoothed[end]:  # A → B (short) → A
                    for j in range(start, end):
                        smoothed[j] = smoothed[start - 1]  # overwrite with A
        else:
            i += 1

    return smoothed


def calculate_clusters(
        method,
        coordinates,
        n_clusters,
        distance_threshold=0.7
):
    """
    The trajectory of the particle is divided into diffusive and non-diffusive sections.
    Points not vibrating are marked as -1 in the classification array.

    Args:
        method      (str):        Clustering method:s 'K-means' or 'Spectral'.
        coordinates (np.ndarray): Input cartesian coordinates for clustering.
        n_clusters  (int):        Number of clusters to create.

    Returns:
        tuple: centers in cartesian coordinates, classification labels
    """
    # Select classifier
    clustering = select_classifier(method, n_clusters)

    # Get labels and centers
    classification = clustering.fit_predict(coordinates)

    centers = np.zeros((n_clusters, 3))
    distances_to_center = np.zeros(len(coordinates))
    for i in range(n_clusters):
        positions = np.where(classification == i)[0]
        centers[i] = np.mean(coordinates[positions], axis=0)
        distances_to_center[positions] = np.linalg.norm(coordinates[positions] - centers[i], axis=1)

    # Mark diffusive points as -1
    classification[distances_to_center > distance_threshold] = -1

    # Smooth classification to remove short excursions
    classification = smooth_classification(classification)
    
    return centers, classification


def update_md_and_stc_info(
        md_info,
        stc_info,
        center_frac_coord,
        ref_frac_coord,
        int_frac_coord,
        center_idx,
        atom_idx,
        is_at_center,
        cell
):
    """Update both md_info (per atom) and stc_info (per reference lattice position)
    based on a center of vibration.

    Args:
        md_info           (dict):    Atom-centered dictionary with lattice/interstitial info.
        stc_info          (dict):    Dictionary with occupancy info on the stoichiometric structure.
        center_frac_coord (ndarray): 3D coordinates of the vibration center (fractional).
        ref_frac_coord    (ndarray): Reference lattice positions (Nx3) in fractional coordinates.
        center_idx        (int):     Index of the center for this atom.
        atom_idx          (int):     Atom index.
        is_at_center      (ndarray): Array (n_steps,) indicating whether it is at the center at each timestep.

    Returns:
        tuple: Updated (md_info, stc_info) dictionaries.
    """
    # Combine lattice and interstitial positions
    all_frac_coord = np.vstack([ref_frac_coord, int_frac_coord])

    # Compute distance to all reference positions with PBC
    diff_frac = all_frac_coord - center_frac_coord

    # Apply pbc
    while np.any(diff_frac > 0.5):
        diff_frac[diff_frac > 0.5] -= 1
    while np.any(diff_frac < -0.5):
        diff_frac[diff_frac < -0.5] += 1

    # Convert to cartesian distances
    diff_cart = diff_frac @ cell

    # Get distances and closest index
    dists       = np.linalg.norm(diff_cart, axis=1)
    closest_idx = np.argmin(dists)

    # Classify center as lattice or interstitial
    if closest_idx < len(ref_frac_coord):
        category = "lattice"
        atom_key = f"atom-{closest_idx}"
    else:
        category = "interstitial"
        atom_key = f"int-{closest_idx - len(ref_frac_coord)}"

    # Update md_info
    md_info[f'atom-{atom_idx}'][category][f'center-{center_idx}'] = {
        'fractional_coordinates': center_frac_coord.tolist(),
        'when': is_at_center.tolist()
    }

    # Accumulate occupancy over atoms: sum previous + current occupation
    if category == 'lattice':
        prev_when = np.array(stc_info[atom_key]['when'], dtype=int)
        stc_info[atom_key]['when'] = (prev_when + is_at_center).tolist()

    return md_info, stc_info


def unique_rows(
        coords,
        tol=1e-4
):
    """Remove duplicate fractional coordinates."""
    uniq = []
    for v in coords:
        if not any(np.allclose(v, u, atol=tol) for u in uniq):
            uniq.append(v)
    return uniq


def get_interstitials(
        ref_path
):
    """
    Get interstitial fractional coordinates using Voronoi analysis.

    Args:
        ref_path (str): Path to the reference structure directory containing POSCAR_pristine.

    Returns:
        list: Array of fractional coordinates of interstitial sites.
    """

    poscar_path = f'{ref_path}/POSCAR_interstitials'
    if os.path.exists(poscar_path):
        # Extract fractional coordinates
        int_frac_coords = load_frac_sites(poscar_path)

    else:
        gen = VoronoiInterstitialGenerator(
            clustering_tol=0.9,
            min_dist=0.9
        )

        # We use the pristine structure with 0K converged lattice as reference
        # Find the POSCAR file in the ref_path
        import glob
        pristine_files = glob.glob(f'{ref_path}/POSCAR-*')
        if not pristine_files:
            pristine_files = glob.glob(f'{ref_path}/POSCAR_pristine')
        if not pristine_files:
            raise FileNotFoundError(f"No reference POSCAR found in {ref_path}")
        pristine_path = pristine_files[0]
        
        ref_structure = Structure.from_file(pristine_path)

        # Use a dummy species H to represent interstitial atoms
        # _get_candidate_sites returns triples: (site, multiplicity, equiv_fpos)
        int_frac_coords = []
        for site, multiplicity, equiv_fpos in gen._get_candidate_sites(ref_structure):
            # equiv_fpos contains all symmetry-related sites
            for fp in equiv_fpos:
                int_frac_coords.append(fp)

        # Build structure for saving
        inter_struct = Structure(
            lattice=ref_structure.lattice,
            species=["H"] * len(int_frac_coords),
            coords=int_frac_coords,
            coords_are_cartesian=False
        )

        # Save POSCAR
        Poscar(inter_struct).write_file(poscar_path)

    return np.array(int_frac_coords)


def load_frac_sites(
        poscar_path
):
    """Load fractional coordinates from a POSCAR file.

    Args:
        poscar_path (str): Path to the POSCAR file.

    Returns:
        ndarray: Array of fractional coordinates.
    """
    # Load original POSCAR_pristine
    structure = Poscar.from_file(poscar_path).structure

    # Extract fractional coordinates
    return [site.frac_coords for site in structure.sites]


def load_trajectory(
        md_path,
        step_skip=2,
        step_equiv=500,
        data_format='parsed',
        filename='CsPbIBr.traj'
):
    """Load MD trajectory and extract fractional and cartesian coordinates.

    Args:
        md_path    (str): Path to the MD trajectory file.
        step_skip  (int): Steps to skip when analyzing trajectory (default 20).
        step_equiv (int): Frames to skip for equilibration (diffusion only).
        data_format (str): 'raw' to return ASE trajectory object, 'parsed' to return coordinates.

    Returns:
        tuple: (md_frac_coords, md_cart_coords, cell, md_steps, md_atoms)
            md_frac_coords (ndarray): Fractional coordinates from MD trajectory.
            md_cart_coords (ndarray): Cartesian coordinates from MD trajectory.
            cell           (ndarray): Cell vectors used in the simulation.
            md_steps       (int):     Number of steps in the MD trajectory after equilibration.
            md_atoms       (int):     Number of atoms in the MD trajectory.
    """
    # Read trajectory (as in convergence analysis)
    print("\nLoading trajectory}...")
    full_traj = read(f"{md_path}/{filename}", index=f'::{step_skip}')
    print(f"Loaded {len(full_traj)} frames (stride={step_skip}).")

    unequiv_traj = full_traj[:step_equiv]
    equiv_traj = full_traj[step_equiv:]  # Remove equilibration frames

    # Safety check: if step_equiv is too large, reduce it to leave at least 10% of frames as production
    if len(equiv_traj) == 0:
        safe_equiv = max(0, int(len(full_traj) * 0.5))
        print(f"WARNING: step_equiv={step_equiv} >= total frames ({len(full_traj)}).")
        print(f"         Reduciendo a {safe_equiv} frames de equilibración (50% del total).")
        print(f"         Para deshabilitar el skip de equilibración, usa step_equiv=0.")
        step_equiv   = safe_equiv
        unequiv_traj = full_traj[:step_equiv]
        equiv_traj   = full_traj[step_equiv:]

    print(f"Removed {step_equiv} equilibration steps.")

    md_steps = len(equiv_traj)
    md_atoms = len(equiv_traj[0])

    # Load simulation parameters
    with open(f'{md_path}/simulation-data.json', 'r') as f:
        params = json.load(f)

    # Simulation parameters
    sim_timestep  = params['timestep'] * params['nblock']
    traj_timestep = sim_timestep  * step_skip
    equiv_time    = traj_timestep * step_equiv / 1000  # from fs to ps
    prod_time     = traj_timestep * md_steps   / 1000  # from fs to ps
    print(f"Equilibration: {equiv_time:.2f} ps, production: {prod_time:.2f} ps, effective timestep: {traj_timestep} fs.")

    if data_format == 'raw':
        return equiv_traj, unequiv_traj, traj_timestep, md_steps, md_atoms

    elif data_format == 'parsed':
        # Use the cell from the first frame after equilibration as fixed reference
        cell = equiv_traj[0].get_cell()[:]
        print(f"Reference cell vectors (fixed from step {step_equiv}):\n{cell}")

        # Extract fractional coordinates from trajectory and convert to cartesian
        md_frac_coords = np.array([atoms.get_scaled_positions(wrap=True) for atoms in equiv_traj])
        md_cart_coords = get_cartesian_coordinates(md_frac_coords, cell)
        return md_frac_coords, md_cart_coords, cell, md_steps, md_atoms

    else:
        sys.exit(f"Data format {data_format} not recognized.")


def md_analysis(
        md_path,
        ref_path,
        step_skip=2,
        step_equiv=500
):
    """Compute both md_info (per atom) and stc_info (per lattice position)
    from MD trajectory and reference structure.

    Args:
        md_path    (str): Path to the MD trajectory file.
        ref_path   (str): Name of the reference structure.
        step_skip  (int): Steps to skip when analyzing trajectory (default 20).
        step_equiv (int): Frames to skip for equilibration (diffusion only).

    Returns:
        tuple: (md_info, stc_info)
            md_info (dict): Atom-centered dictionary with lattice/interstitial info.
            stc_info (dict): Lattice position dictionary with occupancy info (vacancies = when==0).
    """
    # Read trajectory
    md_frac_coords, md_cart_coords, cell, md_steps, md_atoms = load_trajectory(md_path, step_skip, step_equiv)

    # Load reference fractional coordinates
    import glob
    pristine_files = glob.glob(f'{ref_path}/POSCAR-*')
    if not pristine_files:
        pristine_files = glob.glob(f'{ref_path}/POSCAR_pristine')
    if not pristine_files:
        raise FileNotFoundError(f"No reference POSCAR found in {ref_path}")
    pristine_path = pristine_files[0]
        
    ref_frac_coord = load_frac_sites(pristine_path)
    ref_atoms = len(ref_frac_coord)

    # Get interstitial fractional coordinates
    int_frac_coord = get_interstitials(ref_path)

    # Initialize md_info and stc_info
    md_info  = {f'atom-{i}': {'lattice': {}, 'interstitial': {}, 'diffusion': {}} for i in range(md_atoms)}
    stc_info = {f'atom-{i}': {'fractional_coordinates': ref_frac_coord[i].tolist(),
                              'when': [0] * md_steps} for i in range(ref_atoms)}

    # Loop over all particles in md
    centers = []
    diffusion = []
    for atom_idx in range(md_atoms):
        md_cart_coords_idx = md_cart_coords[:, atom_idx]  # shape (n_steps, 3)
        md_frac_coords_idx = md_frac_coords[:, atom_idx]

        n_clusters = calculate_silhouette('K-means', md_cart_coords_idx)

        if n_clusters > 1:
            centers_cart_coord, classification = calculate_clusters('K-means', md_cart_coords_idx, n_clusters)

            diffusion.append(md_cart_coords_idx[classification == -1])

            for center_idx, center_cart_coord in enumerate(centers_cart_coord):
                # Convert center coordinates to fractional coordinates
                center_frac_coord = center_cart_coord @ np.linalg.inv(cell)

                # Convert to ints
                is_at_center = (classification == center_idx).astype(int)

                centers.append(center_cart_coord)
                # Update info
                md_info, stc_info = update_md_and_stc_info(
                    md_info, stc_info,
                    center_frac_coord,
                    ref_frac_coord,
                    int_frac_coord,
                    center_idx, atom_idx,
                    is_at_center, cell
                )
            
            # Append info on diffusion
            if np.any(classification == -1):
                # Convert to ints
                is_at_center = (classification == -1).astype(int)
                
                # Update md_info
                md_info[f'atom-{atom_idx}']['diffusion'][f'center-{center_idx}'] = {
                    'when': is_at_center.tolist()
                }
            
        else:
            # Single center: use first coordinate
            center_frac_coord = mean_frac_coordinates(md_frac_coords_idx)
            center_idx = 0
            is_at_center = np.ones(md_steps, dtype=int)

            # Update info
            md_info, stc_info = update_md_and_stc_info(
                md_info, stc_info,
                center_frac_coord,
                ref_frac_coord,
                int_frac_coord,
                center_idx, atom_idx,
                is_at_center, cell
            )

    return md_info, stc_info, centers, diffusion


def get_defect_evolution(
        md_info,
        stc_info
):
    """
    Extracts time-series data for vacancies, interstitials, and diffusing atoms.
    
    Args:
        md_info (dict): Atom-centered data (tracking where each atom is).
        stc_info (dict): Site-centered data (tracking occupancy of lattice sites).
        
    Returns:
        tuple: (vacancies_array, interstitials_array, diffusion_array)
               Each is a numpy array of shape (n_steps,)
    """
    
    # Determine number of timesteps
    # We grab the length of the 'when' list from the first entry in stc_info
    first_key = next(iter(stc_info))
    n_steps = len(stc_info[first_key]['when'])
    
    # Initialize counters
    vacancies_t     = np.zeros(n_steps, dtype=int)
    interstitials_t = np.zeros(n_steps, dtype=int)
    diffusion_t     = np.zeros(n_steps, dtype=int)

    # Iterate through every reference Lattice Site.
    # If a site has 0 atoms assigned to it at step t, it is a vacancy.
    for site_key, site_data in stc_info.items():
        # occupation is an array of ints (0, 1, or >1)
        occupation = np.array(site_data['when'])
        
        # Identify where occupation is 0
        is_vacant = (occupation == 0).astype(int)
        
        # Add to total count
        vacancies_t += is_vacant

    # Iterate through every atom
    # Check if the atom is currently assigned to an 'interstitial' center
    for atom_key, atom_data in md_info.items():
        # Sum up all interstitial occurrences for this atom
        for center_key, center_data in atom_data['interstitial'].items():
            interstitials_t += np.array(center_data['when']).astype(int)

    # Iterate through every atom
    # Check if the atom was flagged as diffusing
    for atom_key, atom_data in md_info.items():
        # Sum up all diffusion occurrences for this atom
        for center_key, center_data in atom_data['diffusion'].items():
            diffusion_t += np.array(center_data['when']).astype(int)
    return vacancies_t, interstitials_t, diffusion_t
