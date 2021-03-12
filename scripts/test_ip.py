import numpy as np
import pyscf
from pyscf import gto,dft,scf
from pyscf.lib import num_threads
import scipy
from ase import Atoms
from ase.io import read
import pylibnxc
import sys
import pickle
from dpyscf.losses import *
basis = '6-311++G(3df,2pd)'
# basis = '6-311+G*'
# num_threads(1)
try:
    atoms = read('../data/ip_atoms.traj',':')
except:
    atoms = read('../../data/ip_atoms.traj',':')
# systems = [103, 14, 23, 5, 10, 79, 27, 105] #Validation
# atoms = [atoms[s] for s in systems]

# systems = [103, 14] #Validation
# atoms = [atoms[s] for s in systems]
N = len(atoms)

spins = {
    'Al': 1,
    'B' : 1,
    'Li': 1,
    'Na': 1,
    'Si': 2 ,
    'Be':0,
    'C': 2,
    'Cl': 1,
    'F': 1,
    'H': 1,
    'N': 3,
    'O': 2,
    'P': 3,
    'S': 2
}

ips = []
eas = []
for a in atoms:
    print(a)
    energies_predicted = []
    
    for spin, charge in zip(a.info['spin'],[1, 0, -1]):
        pos = a.positions
        this_basis = basis
        spec = a.get_chemical_symbols()
        mol_input = [[s, p] for s, p in zip(spec, pos)]
#         spin = a.info['spin']
        
        mol = gto.M(atom=mol_input, basis=this_basis, spin=spin, charge=charge)
  
        if spin == 0:
            method = pylibnxc.pyscf.RKS
        else:
            method = pylibnxc.pyscf.UKS

        mol.verbose=4
        if sys.argv[2] == 'nxc':
            mf = method(mol, nxc=sys.argv[1], nxc_kind='grid')
        elif sys.argv[2] == 'xc':
            mf = method(mol)
            mf.xc = sys.argv[1]

        if len(sys.argv) > 3:
            mf.grids.level= int(sys.argv[3])
        else:
            mf.grids.level = 9

        mf.kernel()
        energies_predicted.append(mf.e_tot)
        
    eas.append(-energies_predicted[2]+energies_predicted[1])
    ips.append(-energies_predicted[1]+energies_predicted[0])


np.save(sys.argv[1] + '_ip.npy', ips)
np.save(sys.argv[1] + '_ea.npy', eas)