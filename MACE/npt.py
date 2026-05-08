import os
import glob
import sys
import argparse
import io
import json
from tqdm import tqdm
from mace.calculators            import mace_mp
from ase                         import units
from ase.md.npt                  import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.vasp                 import read_vasp, write_vasp
from ase.io.trajectory           import TrajectoryWriter

class TeeLogger:
    def __init__(self, buffer):
        self.buffer = buffer
    def write(self, data):
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
    args = parser.parse_args()

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

    model_load_path = os.path.join(BASE_DIR, 'mace-mpa-0-medium.model')

    # Simulation parameters
    dispersion  = False
    temperature = 1200  # K
    timestep_fs = 1     # fs
    pressure_gpa= 0     # GPa
    ttime_fs    = 50    # fs
    ptime_fs    = 500   # fs
    n_steps     = args.steps

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

        # Load the pre-trained model
        atoms.calc = mace_mp(model=model_load_path, device=args.device, dispersion=dispersion, default_dtype='float64')

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
        tee_logger = TeeLogger(log_buffer)

        # Perform the NPT molecular dynamics
        dyn = NPT(atoms, timestep=timestep, temperature_K=temperature, ttime=ttime, pfactor=pfactor, 
                  externalstress=pressure, logfile=tee_logger, loginterval=10)

        # In-memory trajectory saving
        trajectory_frames = []
        def append_frame():
            trajectory_frames.append(atoms.copy())
            
        dyn.attach(append_frame, interval=1)

        # Run dynamic
        print(f"Running {n_steps} steps of NPT dynamics...")
        pbar = tqdm(total=n_steps, desc=f"MD {material}", unit="step")
        
        def update_pbar():
            pbar.update(1)
            
        dyn.attach(update_pbar, interval=1)
        dyn.run(n_steps)
        pbar.close()
        
        print("Dynamics completed. Saving results...")

        # Write trajectory to disk
        traj_writer = TrajectoryWriter(filename, mode='w')
        for frame in trajectory_frames:
            traj_writer.write(frame)
        traj_writer.close()
        print(f"Trajectory saved to {filename}")

        # Write log to disk
        with open(logname, 'w') as f:
            f.write(log_buffer.getvalue())
        print(f"Log saved to {logname}")

        # Write final structure
        write_vasp(contcar, trajectory_frames[-1], direct=True)
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
