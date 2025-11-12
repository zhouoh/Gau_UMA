from ase.build import molecule
import ase
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import numpy as np
import pickle
import socket
import struct
import io
import json
import logging
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


MODEL_NAME = "uma-s-1p1"
PORT = 50423


bohr2angstrom = 0.529177249
angstrom2bohr = 1.0/bohr2angstrom
ev2hartree = 0.036749322

def send_object(sock, obj):
    data = pickle.dumps(obj)
    message_header = struct.pack('Q', len(data))
    sock.sendall(message_header)
    sock.sendall(data)
    logging.info("Sent result to client.")

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def receive_calculation_job(sock):
    # 接收元数据帧
    header_len_bytes = recvall(sock, 8)
    if not header_len_bytes: return None, None
    header_len = struct.unpack('Q', header_len_bytes)[0]
    header_bytes = recvall(sock, header_len)
    if not header_bytes: return None, None
    metadata = json.loads(header_bytes.decode('utf-8'))
    
    # 接收数据帧
    payload_len_bytes = recvall(sock, 8)
    if not payload_len_bytes: return None, None
    payload_len = struct.unpack('Q', payload_len_bytes)[0]
    payload_bytes = recvall(sock, payload_len)
    if not payload_bytes: return None, None
    xyz_content = payload_bytes.decode('utf-8')
    
    return metadata, xyz_content

def Energy(atoms):
    """
    计算分子的能量和力。
    """
    energy = atoms.get_potential_energy() * ev2hartree
    return energy

def Force(atoms):
    """
    计算分子的力。
    """
    forces = -1 * atoms.get_forces() * ev2hartree / angstrom2bohr
    return forces

def get_hessian_numeric(atoms, h = 1e-3) -> np.ndarray:

    num_atoms = atoms.get_number_of_atoms()
    original_positions = atoms.get_positions()
    print(original_positions)
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

# --- 主函数 ---
def main():
    settings = InferenceSettings(
    tf32=False,
    activation_checkpointing=False,
    merge_mole=False,
    external_graph_gen=False,
    internal_graph_gen_version=2,
    )
    predictor = pretrained_mlip.get_predict_unit(MODEL_NAME, device="cuda",inference_settings=settings)
    mol = molecule("H2O")
    calc = FAIRChemCalculator(predictor, task_name="omol")
    mol.set_calculator(calc)
    mol.get_potential_energy()  # 预热模型

    server_address = ('localhost', PORT)
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(server_address)
        server_socket.listen(1)
        logging.info(f"Listening on port {PORT}...")

        while True:
            conn, addr = server_socket.accept()
            with conn:
                logging.info(f"Received connection from {addr}")
                
                metadata, xyz_content = receive_calculation_job(conn)
                
                if metadata and xyz_content:
                    logging.info(f"Received job with metadata: {metadata}")
                    atoms = ase.io.read(io.StringIO(xyz_content), format='xyz')
                    atoms.set_calculator(calc)
                    charge = metadata.get('charge', 0)
                    multiplicity = metadata.get('multiplicity', 1)
                    atoms.info.update({'charge': charge, 'spin': multiplicity})
                    atoms.set_pbc(False)
                    
                    calc_type = metadata.get('calc_type')
                    result = {}
                    if calc_type == 'energy':
                        energy = Energy(atoms)
                        result['energy'] = energy
                        logging.info("Energy calculation completed.")
                    elif calc_type == 'force':
                        energy = Energy(atoms)
                        result['energy'] = energy
                        forces = Force(atoms)
                        result['forces'] = forces
                        logging.info("Force calculation completed.")
                    elif calc_type == 'hessian':
                        energy = Energy(atoms)
                        result['energy'] = energy
                        forces = Force(atoms)
                        result['forces'] = forces
                        hessian = Hessian(atoms)
                        result['hessian'] = hessian
                        logging.info("Hessian calculation completed.")
                    else:
                        logging.warning(f"Unknown calculation type: {calc_type}")
                        continue
                
                send_object(conn, result)

if __name__ == "__main__":
    main()


                


