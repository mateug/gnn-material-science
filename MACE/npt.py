from mace.calculators            import mace_mp
from ase                         import units
from ase.md.npt                  import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.vasp                 import read_vasp, write_vasp, write_vasp_xdatcar
from ase.io.trajectory           import Trajectory

"""
Podemos definir:
    # Initial settings
    path_to_structure: dónde está nuestra estructura inicial.
    filename: con qué nombre guardamos la dinámica.
    logname: con qué nombre guardamos los logs de la simulación.
    model_load_path: modelo a usar (el que pongo es el más potente para inorgánicos).
    device: dónde correrlo (cuda, que es gpu, o cpu).
    
    # NPT parameters
    dispersion: si queremos dispersiones de largo alcance (como fuerzas van der Waals), en principio no necesario para perovskitas.
    temperature: temperatura de la simulación (a 1200K debería haber difusión iónica).
    timestep: paso de tiempo (entre 1 y 1.5 fs está bien).
    pressure: presión externa (a cero en principio).
    ttime, ptime: constantes del algoritmo NPT, que tocaremos si volumen, temperatura o presión no convergen establemente.
    n_steps: número de pasos en la simulación (50000 con timestep=1 son 50 ps).

En la carpeta veremos:
    filename: fichero que podemos leer con python con toda la simulación. 
    logname: fichero con posibles errores/warning durante la dinámica.
    CONTCAR: estructura final, tras toda la dinámica.
"""

# Initial settings
path_to_structure = 'POSCAR-CsPbBr-supercell-4x4x4'
filename          = 'CsPbIBr.traj'  # Saving output
logname           = 'npt.log'  # Saving log
model_load_path   = 'mace-mpa-0-medium.model'  # ASE model
device            = 'cpu'  # GPU acceleration

# NPT parameters
dispersion  = False
temperature = 1200  # K
timestep    = 1  # fs
pressure    = 0  # external pressure, GPa
ttime       = 50  # thermostat time constant, fs
ptime       = 500  # barostat time constant, fs
n_steps     = 50000

# Read structure
atoms = read_vasp(file=path_to_structure)

# Load the pre-trained model
atoms.calc = mace_mp(model=model_load_path, device=device, dispersion=dispersion, default_dtype='float64')

# Set units
timestep *= units.fs
ttime    *= units.fs
ptime    *= units.fs
pressure /= 160.21766208  # GPa -> eV/Å³

pfactor = units.GPa * ptime**2

# Initialize velocities
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

# Perform the NPT molecular dynamics (Full Martyna-Tobias-Klein dynamics)
dyn = NPT(atoms, timestep=timestep, temperature_K=temperature, ttime=ttime, pfactor=pfactor, externalstress=pressure, trajectory=filename, logfile=logname)

# Run dynamic
dyn.run(n_steps)

# Write trajectory as XDATCAR
traj = Trajectory(filename)
write_vasp('CONTCAR', traj[-1], direct=True)  # Save the final geometrical structure as CONTCAR
