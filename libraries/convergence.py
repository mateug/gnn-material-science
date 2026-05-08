import numpy as np
import matplotlib.pyplot as plt
import libraries.dynamics as dyn
import json

from ase.io                               import read
from ase                                  import units
from pymatgen.io.ase                      import AseAtomsAdaptor
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer


def plot_and_fit(
        equiv_x,
        unequiv_x,
        equiv_y,
        unequiv_y,
        md_path,
        data_name,
        xlabel='t (ps)',
        ylabel=''
):
    """
    Plot a quantity y vs x.
    - Data before n_skip is plotted with alpha=0.5 (Equilibration).
    - Data after n_skip is plotted normally (Production).
    - Statistics (mean/std) are calculated ONLY on Production data.
    
    Returns: fit coefficients (mean, std) of the production phase.
    """
    colors = {
        'pressure': 'tab:blue',
        'temperature': 'tab:green',
        'volume': 'tab:purple',
        'total-energy': 'tab:olive',
        'potential-energy': 'tab:red',
        'kinetic-energy': 'tab:orange'
    }

    # Calculate stats on Production only
    mean = np.mean(equiv_y)
    std  = np.std(equiv_y)
    
    # Plot equilibration (faint) and production (solid)
    plt.plot(unequiv_x, unequiv_y, color=colors.get(data_name, 'k'), alpha=0.5, label='Equilibration')
    plt.plot(equiv_x,   equiv_y,   color=colors.get(data_name, 'k'), alpha=1.0, label='Production')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{data_name} convergence')
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{md_path}/{data_name}-convergence.pdf', dpi=100)
    plt.show()
    plt.close()
    return mean, std


def extract_thermo_data(
        traj,
        traj_timestep,
        time_offset=0
):
    """
    Extracts time, temperature, pressure, and volume data from an ASE trajectory.

    Args:
        traj: ASE trajectory object.
        traj_timestep (float): Time step between frames in fs.

    Returns:
        times       (list): List of times in ps.
        temperatures(list): List of temperatures in K.
        pressures   (list): List of pressures in GPa.
        volumes     (list): List of volumes in Å³.
    """
    times = []
    volumes = []
    pressures = []
    temperatures = []
    for i, atoms in enumerate(traj):
        # Time (ps)
        t_ps = i * traj_timestep / 1000 + time_offset
        times.append(t_ps)

        # Volume (Å³)
        vol = atoms.get_volume()
        volumes.append(vol)

        # Temperature
        T = atoms.get_temperature()
        temperatures.append(T)

        # Virial (from stress tensor, check if available)
        stress = atoms.get_stress()
        virial_pressure = -np.mean(stress[:3])  # simple average of diagonal

        # Kinetic pressure = (N * kB * T) / V
        kinetic_pressure = (len(atoms) * units.kB * T) / vol

        # Total Pressure (eV/Å³ -> GPa conversion factor ~160.21766)
        P_GPa = (virial_pressure + kinetic_pressure) * 160.21766208
        pressures.append(P_GPa)
    return times, temperatures, pressures, volumes


def analyze_convergence_and_diffusion(
        md_path='.',
        filename='CsPbIBr.traj',
        step_skip=2,
        step_equiv=500
):
    """
    Performs comprehensive NPT analysis:
    1. Checks convergence of T, P, V (stats and plots).
    2. Calculates Diffusion Coefficient and Ionic Conductivity for all species.
    3. Saves summary to JSON.
    
    Args:
        md_path    (str): Path to the MD trajectory file.
        filename   (str): Name of the trajectory file.
        step_skip  (int): Stride for diffusion analysis (use 1 for max accuracy).
        step_equiv (int): Frames to skip for equilibration (diffusion only).
    """
    traj_file   = f'{md_path}/{filename}'
    json_file   = f'{md_path}/simulation-data.json'
    output_json = f'{md_path}/npt-convergence.json'
    
    print(f"Starting analysis for {traj_file}...")

    # Load simulation parameters
    with open(json_file, 'r') as f:
        params = json.load(f)
    temperature  = params['temperature']

    # Convergence Analysis (T, P, V)
    print("\nAnalyzing convergence (T, P, V)...")
    
    # Read trajectory
    equiv_traj, unequiv_traj, traj_timestep, md_steps, md_atoms = dyn.load_trajectory(md_path, step_skip, step_equiv,
                                                                                data_format='raw', filename=filename)

    # Extract thermo data
    unequiv_times, unequiv_temperatures, unequiv_pressures, unequiv_volumes = extract_thermo_data(unequiv_traj,
                                                                                                  traj_timestep)
    equiv_times,   equiv_temperatures,   equiv_pressures,   equiv_volumes   = extract_thermo_data(equiv_traj,
                                                                                                  traj_timestep,
                                                                                                  time_offset=unequiv_times[-1])

    # Plot and Fit (Passing step_equiv to exclude equilibration from stats/alpha)
    mean_T, std_T = plot_and_fit(
        equiv_times, unequiv_times,
        equiv_temperatures, unequiv_temperatures,
        md_path, data_name='temperature', ylabel='T (K)')
    mean_P, std_P = plot_and_fit(
        equiv_times, unequiv_times,
        equiv_pressures, unequiv_pressures,
        md_path, data_name='pressure',    ylabel='P (GPa)')
    mean_V, std_V = plot_and_fit(
        equiv_times, unequiv_times,
        equiv_volumes, unequiv_volumes,
        md_path, data_name='volume',      ylabel='V (Å³)')
    
    # Diffusion analysis
    print("\nAnalyzing diffusion...")
    
    # Convert to Pymatgen structures
    adaptor = AseAtomsAdaptor()
    structures = [adaptor.get_structure(atoms) for atoms in equiv_traj]
    
    # Detect species
    unique_species = sorted(list(set(structures[0].composition.get_el_amt_dict().keys())))
    print(f"Detected species: {unique_species}")

    diffusion_results = {}

    for specie in unique_species:
        print(f"  > Processing {specie}...")
        try:
            analyzer = DiffusionAnalyzer.from_structures(
                structures=structures,
                specie=specie,
                temperature=temperature,
                time_step=traj_timestep, # Time between analyzed frames
                step_skip=1,  # We already skipped frames during load
                smoothed="max"
            )

            diffusion_results[specie] = {
                'diffusivity_cm2s':      getattr(analyzer, 'diffusivity', 0.0),
                'diffusivity_cm2s_std':  getattr(analyzer, 'diffusivity_std_dev', 0.0),
                'conductivity_mScm':     getattr(analyzer, 'conductivity', 0.0),
                'conductivity_mScm_std': getattr(analyzer, 'conductivity_std_dev', 0.0)
            }
            
            # Extract MSD data
            msd = getattr(analyzer, 'msd', [])

            # time_step in analyzer is stored in fs to ps
            times_ps = np.arange(len(msd)) * traj_timestep / 1000
            plt.plot(times_ps, msd, label=f'MSD ({specie})')
        except Exception as e:
            print(f"\tError analyzing {specie}: {e}")

    plt.xlabel("Time (ps)")
    plt.ylabel(r"MSD ($\AA^2$)")
    plt.xlim(left=-0.1)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{md_path}/diffusion.pdf', dpi=50)
    plt.show()
    
    # Save results
    print("\nSaving results...")
    
    data = {
        'temperature (K)': {'mean': mean_T, 'std': std_T},
        'pressure (GPa)':  {'mean': mean_P, 'std': std_P},
        'volume (A^3)':    {'mean': mean_V, 'std': std_V},
        'diffusion': diffusion_results
    }
    
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Results saved to {output_json}")
    
    # Print summary to console
    print("-" * 40)
    print("Summary (stats calculated on production frames only)")
    print(f"T (K):   {mean_T:.2f} +/- {std_T:.2f}")
    print(f"P (GPa): {mean_P:.2f} +/- {std_P:.2f}")
    print(f"V (A^3): {mean_V:.2f} +/- {std_V:.2f}")
    print("Diffusion:")
    for sp, vals in diffusion_results.items():
        print(f"  {sp}: D={vals['diffusivity_cm2s']:.2e} cm²/s, σ={vals['conductivity_mScm']:.2f} mS/cm")
    print("-" * 40)
