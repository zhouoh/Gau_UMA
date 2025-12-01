import sys
import os
import io
import json
import logging
import numpy as np
from ase.build import molecule
import ase.io
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
import torch


torch.set_num_threads(16) 

# --- Constants ---
MODEL_NAME = "uma-s-1p1"
# Based on the filename GauUMA_CPU.py, we assume the user intends to run this on CPU.
DEVICE = "cpu" 

bohr2angstrom = 0.529177249
angstrom2bohr = 1.0/bohr2angstrom
ev2hartree = 0.036749322

PeriodicTable = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"]

# --- Calculation Functions (from GauUMA_server.py) ---

def Energy(atoms):
    energy = atoms.get_potential_energy() * ev2hartree
    return energy

def Force(atoms):
    forces = -1 * atoms.get_forces() * ev2hartree / angstrom2bohr
    return forces

def get_hessian_numeric(atoms, h = 1e-3) -> np.ndarray:
    num_atoms = atoms.get_number_of_atoms()
    original_positions = atoms.get_positions()
    # print(original_positions) # Suppressed for cleaner output in CLI usage
    hessian_matrix = np.zeros((len(atoms) * 3, len(atoms) * 3))
    for i in range(num_atoms):
        for j in range(3):
            positions_plus = original_positions.copy()
            positions_plus[i, j] += h
            atoms.set_positions(positions_plus)
            force_plus = atoms.get_forces()

            positions_minus = original_positions.copy()
            positions_minus[i, j] -= h
            atoms.set_positions(positions_minus)
            force_minus = atoms.get_forces()
            
            hessian_column = (force_plus.flatten() - force_minus.flatten()) / (2 * h)
            hessian_matrix[:, i * 3 + j] = -hessian_column 

    atoms.set_positions(original_positions)
    return hessian_matrix

def Hessian(atoms) -> np.ndarray:
    """
    计算分子的Hessian矩阵。
    """
    hessian = get_hessian_numeric(atoms, h=1e-3)
    hessian =  hessian * ev2hartree / (angstrom2bohr**2)
    return hessian

# --- IO Functions (from GauUMA_client.py) ---

def save_result(result, output_file, calctype):
    if calctype == 0:  # energy
        with open(output_file, 'w') as f:
            energy = result["energy"]
            #energy, dipole-moment (xyz)t			E, Dip(I), I=1,3			4D20.12
            f.write(f"{energy:20.12e}{0:20.12e}{0:20.12e}{0:20.12e}\n")
    elif calctype == 1:  # force
        with open(output_file, 'w') as f:
            energy = result["energy"]
            forces = result["forces"]
            #energy, dipole-moment (xyz)t			E, Dip(I), I=1,3			4D20.12
            f.write(f"{energy:20.12e}{0:20.12e}{0:20.12e}{0:20.12e}\n") 
            #gradient on atom (xyz)			FX(J,I), J=1,3; I=1,NAtoms			3D20.12
            for i in range(len(forces)):
                f.write(f"{forces[i,0]:20.12e}{forces[i,1]:20.12e}{forces[i,2]:20.12e}\n".replace('e', 'E'))
    elif calctype == 2:  # hessian
        with open(output_file, 'w') as f:
            energy = result["energy"]
            forces = result["forces"]
            hessian = result["hessian"]
            #energy, dipole-moment (xyz)t			E, Dip(I), I=1,3			4D20.12
            f.write(f"{energy:20.12e}{0:20.12e}{0:20.12e}{0:20.12e}\n") 
            #gradient on atom (xyz)			FX(J,I), J=1,3; I=1,NAtoms			3D20.12
            for i in range(len(forces)):
                f.write(f"{forces[i,0]:20.12e}{forces[i,1]:20.12e}{forces[i,2]:20.12e}\n".replace('e', 'E'))
            #polarizability		Polar(I), I=1,6		3D20.12
            for i in range(2):
                f.write(f"{0:20.12e}{0:20.12e}{0:20.12e}\n".replace('e', 'E'))
            #dipole derivatives		DDip(I), I=1,9*NAtoms		3D20.12
            for i in range(3*len(forces)):
                f.write(f"{0:20.12e}{0:20.12e}{0:20.12e}\n".replace('e', 'E'))
            #force constants		FFX(I), I=1,(3*NAtoms*(3*NAtoms+1))/2		3D20.12
            #the Hessian is given in lower triangular form: αij, i=1 to N, j=1 to i. 
            num_coords = hessian.shape[0]
            triangle_indices = np.tril_indices(num_coords)
            hessian_lower = hessian[triangle_indices]
            for i in range(0, len(hessian_lower), 3):
                row = hessian_lower[i:i+3]
                while len(row) < 3:
                    row = np.append(row, 0.0)
                f.write(f"{row[0]:20.12e}{row[1]:20.12e}{row[2]:20.12e}\n".replace('e', 'E'))

# --- Main Logic ---

def main():
    if len(sys.argv) < 4:
        print("Usage: python GauUMA_CPU.py <layer> <InputFile> <OutputFile> ...")
        sys.exit(1)

    input_file = sys.argv[2]
    output_file = sys.argv[3]

    # 1. Parse Input File (from Client)
    calctyp_dict = {0: "energy", 1: "force", 2: "hessian"}
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
            # Handle empty file or bad format gracefully? 
            # Assuming Gaussian always provides valid input per original code.
            NUM_ATOM = int(lines[0].split()[0])
            calctype = int(lines[0].split()[1])
            # calctype_str = calctyp_dict.get(calctype, "energy") # used in client to send to server
            charge = int(lines[0].split()[2])
            multiplicity = int(lines[0].split()[3])
            Geom = np.zeros((NUM_ATOM, 3))
            Elements = []
            for i in range(1, NUM_ATOM + 1):
                line = lines[i].split()
                Elements.append(PeriodicTable[int(line[0]) - 1])
                for j in range(1, 4):
                    Geom[i-1, j-1] = float(line[j])
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # 2. Setup Model (from Server)
    try:
        settings = InferenceSettings(
            tf32=False,
            activation_checkpointing=False,
            merge_mole=False,
            external_graph_gen=False,
            internal_graph_gen_version=2,
        )
        predictor = pretrained_mlip.get_predict_unit(MODEL_NAME, device=DEVICE, inference_settings=settings)
        # We need a dummy mol to initialize the calculator wrapper
        # The server used molecule("H2O"), we can do the same.
        #mol_dummy = molecule("H2O")
        calc = FAIRChemCalculator(predictor, task_name="omol")
        #mol_dummy.set_calculator(calc)
        #mol_dummy.get_potential_energy()  # Warmup
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)

    # 3. Prepare Atoms Object
    # The client created an XYZ string and the server parsed it. 
    # We can skip the string serialization/deserialization and build Atoms directly.
    # However, to be absolutely sure we match the server's behavior (which uses ase.io.read(format='xyz')),
    # we can reconstruct the atoms object carefully.
    
    # Using ASE to build atoms directly is cleaner.
    from ase import Atoms
    # Elements is a list of symbols (e.g., ['C', 'H']). Geom is in Bohrs.
    # ASE expects Angstroms.
    positions_angstrom = Geom * bohr2angstrom
    atoms = Atoms(symbols=Elements, positions=positions_angstrom)
    
    # Set calculator
    atoms.set_calculator(calc)
    
    # Set info (metadata)
    atoms.info.update({'charge': charge, 'spin': multiplicity})
    atoms.set_pbc(False)

    # 4. Perform Calculation
    result = {}
    # Map calctype integer from input file to logic
    # Client mapped: calctyp_dict = {0: "energy", 1: "force", 2: "hessian"}
    
    # Note: Client passed 'calctype_str' to server. Server checked:
    # if calc_type == 'energy'...
    
    if calctype == 0: # energy
        energy = Energy(atoms)
        result['energy'] = energy
        print("Energy calculation completed.")
    elif calctype == 1: # force
        energy = Energy(atoms)
        result['energy'] = energy
        forces = Force(atoms)
        result['forces'] = forces
        print("Force calculation completed.")
    elif calctype == 2: # hessian
        energy = Energy(atoms)
        result['energy'] = energy
        forces = Force(atoms)
        result['forces'] = forces
        hessian = Hessian(atoms)
        result['hessian'] = hessian
        print("Hessian calculation completed.")
    else:
        print(f"Unknown calculation type code: {calctype}")
        sys.exit(1)

    # 5. Output Result (Client logic)
    print(f"SCF Done:  E(UMA) =  {result['energy']}     A.U.")
    save_result(result, output_file, calctype)

if __name__ == "__main__":
    main()
