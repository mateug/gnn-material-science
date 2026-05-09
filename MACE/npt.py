import os
import glob
import sys
import argparse
import io
import json
from tqdm import tqdm
import warnings
from mace.calculators            import mace_mp
from ase                         import units
from ase.md.melchionna           import MelchionnaNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.vasp                 import read_vasp, write_vasp
from ase.io.trajectory           import TrajectoryWriter

class TeeLogger:
    def __init__(self, buffer, print_to_stdout=True):
        self.buffer = buffer
        self.print_to_stdout = print_to_stdout
    def write(self, data):
        if self.print_to_stdout:
            sys.stdout.write(data)
            sys.stdout.flush()
        self.buffer.write(data)
    def flush(self):
        sys.stdout.flush()
        self.buffer.flush()
    def close(self):
        pass

def main():
    parser = argparse.ArgumentParser(description="MACE NPT Simulation Workflow")
    parser.add_argument('--candidates', type=str, default='candidates.txt', help='File with list of candidate materials')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run MACE model (cpu/cuda)')
    parser.add_argument('--steps', type=int, default=50000, help='Number of MD steps')
    parser.add_argument('--progress', type=str, default='log', choices=['log', 'bar'], help='Progress display type: log or bar')
    args = parser.parse_args()

    # Suppress PyTorch weights_only warning triggered by MACE
    warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    candidates_file = os.path.join(ROOT_DIR, args.candidates)
    
    if not os.path.exists(candidates_file):
        print(f"Error: Candidates file {candidates_file} not found.")
        sys.exit(1)

    # Read materials
    materials = []
    with open(candidates_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Extract the first word (material name)
                mat = line.split()[0]
                materials.append(mat)
    
    print(f"Found {len(materials)} materials to process: {materials}")

    if args.device == 'cpu':
        print("Local CPU environment detected. Applying performance optimizations: small model, 2fs timestep, fewer steps.")
        model_load_path = "small"
        timestep_fs = 2     # fs
        n_steps = args.steps // 2 if args.steps >= 2 else args.steps
        supercell_target = 100
    else:
        model_load_path = os.path.join(BASE_DIR, 'mace-mpa-0-medium.model')
        if not os.path.exists(model_load_path):
            model_load_path = "medium"
        timestep_fs = 1     # fs
        n_steps = args.steps
        supercell_target = 150

    # Simulation parameters
    dispersion  = False
    temperature = 1200  # K
    pressure_gpa= 0     # GPa
    ttime_fs    = 50    # fs
    ptime_fs    = 500   # fs

    for material in materials:
        print(f"\n{'='*50}\nProcessing material: {material}\n{'='*50}")

        # Find POSCAR
        search_path = os.path.join(ROOT_DIR, 'input', 'candidates', material, '**', 'POSCAR')
        poscar_files = glob.glob(search_path, recursive=True)
        
        if not poscar_files:
            print(f"Warning: No POSCAR found for {material} in input/candidates/{material}/")
            continue
            
        path_to_structure = poscar_files[0]
        print(f"Found structure at: {path_to_structure}")

        # Setup results directory
        results_dir = os.path.join(BASE_DIR, 'results', material)
        os.makedirs(results_dir, exist_ok=True)

        filename = os.path.join(results_dir, f'{material}.traj')
        logname  = os.path.join(results_dir, 'npt.log')
        contcar  = os.path.join(results_dir, 'CONTCAR')
        sim_data = os.path.join(results_dir, 'simulation-data.json')

        # Read structure
        atoms = read_vasp(file=path_to_structure)

        # Create a supercell if the structure is too small
        multiplier = 1
        if len(atoms) < supercell_target:
            import math
            multiplier = math.ceil((supercell_target / len(atoms)) ** (1/3))
            print(f"Structure is too small ({len(atoms)} atoms). Target is {supercell_target}.")
            print(f"Creating a {multiplier}x{multiplier}x{multiplier} supercell...")
            atoms = atoms * (multiplier, multiplier, multiplier)
        
        print(f"Final structure size for MD: {len(atoms)} atoms")

        # Save the supercell as the reference pristine structure for future defect analysis
        if multiplier > 1:
            pristine_path = os.path.join(results_dir, f'POSCAR-supercell-{multiplier}x{multiplier}x{multiplier}')
        else:
            pristine_path = os.path.join(results_dir, 'POSCAR-unitcell')
        write_vasp(pristine_path, atoms, direct=True)
        print(f"Saved supercell reference to {pristine_path}")

        # Load the pre-trained model
        atoms.calc = mace_mp(model=model_load_path, device=args.device, dispersion=dispersion, default_dtype='float32')

        # Set units for NPT
        timestep = timestep_fs * units.fs
        ttime    = ttime_fs * units.fs
        ptime    = ptime_fs * units.fs
        pressure = pressure_gpa / 160.21766208  # GPa -> eV/Å³
        pfactor  = units.GPa * ptime**2

        # Initialize velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

        # We will use an in-memory buffer for the log, but print to stdout as well
        log_buffer = io.StringIO()
        # En modo 'bar', los logs por paso se silencian (van solo al fichero)
        tee_logger = TeeLogger(log_buffer, print_to_stdout=(args.progress == 'log'))

        # Perform the NPT molecular dynamics
        dyn = MelchionnaNPT(atoms, timestep=timestep, temperature_K=temperature, ttime=ttime, pfactor=pfactor, 
                            externalstress=pressure, logfile=tee_logger, loginterval=10)

        # Streaming trajectory saving to prevent memory leaks
        traj_writer = TrajectoryWriter(filename, mode='w')
        def append_frame():
            traj_writer.write(atoms)
            
        dyn.attach(append_frame, interval=10)

        # Run dynamic
        import torch
        device_status = "CUDA (GPU)" if args.device == "cuda" and torch.cuda.is_available() else "CPU"
        if args.device == "cuda" and not torch.cuda.is_available():
            device_status = "CPU (WARNING: CUDA requested but not found)"
        
        print(f"Running {n_steps} steps of NPT dynamics... [Hardware: {device_status}]")

        if args.progress == 'bar':
            pbar = tqdm(total=n_steps, desc=f"MD {material}", unit="step", dynamic_ncols=True)
            def update_pbar():
                pbar.update(1)
            dyn.attach(update_pbar, interval=1)

        dyn.run(n_steps)

        if args.progress == 'bar':
            pbar.close()

        print("Dynamics completed. Saving results...")
        traj_writer.close()
        print(f"Trajectory saved to {filename}")

        # Write log to disk
        with open(logname, 'w') as f:
            f.write(log_buffer.getvalue())
        print(f"Log saved to {logname}")

        # Write final structure
        write_vasp(contcar, atoms, direct=True)
        print(f"Final structure saved to {contcar}")

        # Write simulation-data.json
        params = {
            "temperature": temperature,
            "timestep": timestep_fs,
            "nblock": 1
        }
        with open(sim_data, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"Simulation parameters saved to {sim_data}")

if __name__ == '__main__':
    main()
