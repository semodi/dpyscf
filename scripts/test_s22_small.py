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
basis = '6-311++G**'
# basis = '6-31G'
# num_threads(1)
try:
    atoms = read('../data/s22.traj',':')
except:
    atoms = read('../../data/s22.traj',':')
# systems = [103, 14, 23, 5, 10, 79, 27, 105] #Validation
# atoms = [atoms[s] for s in systems]

# systems = [103, 14] #Validation
# atoms = [atoms[s] for s in systems]
N = len(atoms)
energies = []
for a in atoms:
    print(a)


    
   
    this_basis = basis
    
    pos = a.positions
    spec = a.get_chemical_symbols()
    
    
    mol_input = [[s, p] for s, p in zip(spec, pos)]
    mol12 = gto.M(atom=mol_input, basis=this_basis)
    
    sep = a.info['sep']
    mol_input1 = [[s, p] for s, p in zip(spec[:sep], pos[:sep])] + \
                 [['X:'+s, p] for s, p in zip(spec[sep:], pos[sep:])]
    
    mol1 = gto.M(atom=mol_input1, basis=this_basis)
    
    mol_input2 = [['X:'+s, p] for s, p in zip(spec[:sep], pos[:sep])] + \
                 [[s, p] for s, p in zip(spec[sep:], pos[sep:])]
    
    mol2 = gto.M(atom=mol_input2, basis=this_basis)
    
    method = pylibnxc.pyscf.RKS

    energy = 0
    for mol, pref in zip([mol12, mol1, mol2],[1,-1,-1]):
        mol.verbose=4
    #     mf = method(mol, nxc='MGGA_trial4', nxc_kind='grid')
        if sys.argv[2] == 'nxc':
            mf = method(mol, nxc=sys.argv[1], nxc_kind='grid')
        elif sys.argv[2] == 'xc':
            mf = method(mol)
            mf.xc = sys.argv[1]

        if len(sys.argv) > 3:
            mf.grids.level= int(sys.argv[3])
        else:
            mf.grids.level = 3
    #     mf.grids.level=1
        mf.kernel()
        energy += pref* mf.e_tot
    
    energies.append(energy)

np.save(sys.argv[1] + '_s22.npy', energies)
    