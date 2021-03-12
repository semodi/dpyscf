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

# num_threads(1)
try:
    atoms = read('../data/haunschild_g2/g2_97.traj',':')
except:
    atoms = read('../../data/haunschild_g2/g2_97.traj',':')
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
atoms += [Atoms(a, info={'spin':spins[a]}) for a in np.unique([s for a in atoms for s in a.get_chemical_symbols() ])]

energies_predicted = []
atoms_predicted = {}
energies_baseline = []
pred_dict = {}
dm_predicted = []
grids = []
mols = []

for a in atoms:
    print(a)


    pos = a.positions
    this_basis = basis
    spec = a.get_chemical_symbols()
    mol_input = [[s, p] for s, p in zip(spec, pos)]
    if len(pos) == 1:
        spin = spins[spec[0]]
    else:
        if a.info['openshell']:
            spin = 2
        else:
            spin = 0
        
    try:
        mol = gto.M(atom=mol_input, basis=this_basis,spin=spin)
    except Exception:
        spin =1
        mol = gto.M(atom=mol_input, basis=this_basis,spin=spin)

    if spin == 0:
        method = pylibnxc.pyscf.RKS
    else:
        method = pylibnxc.pyscf.UKS

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
        mf.grids.level = 9
#     mf.grids.level=1
    mf.kernel()
    dm = mf.make_rdm1()
    if dm.ndim == 2:
        dm = np.stack([dm,dm])*0.5
    dm_predicted.append(dm)
    
    if len(pos) == 1:
        atoms_predicted[a.get_chemical_symbols()[0]] = mf.e_tot
    else:
        energies_predicted.append(mf.e_tot)

ae_energies = []
for energy, a in zip(energies_predicted, atoms):
    symbols = ''.join(a.get_chemical_symbols())
    pred_dict = {symbols: energy}
    pred_dict.update(atoms_predicted)
    ae_energies.append(atomization_energies(pred_dict)[symbols])


np.save('{}_g2_ae.npy'.format(sys.argv[1]),ae_energies)
with open('{}_g2.dm'.format(sys.argv[1]),'wb') as file:
    pickle.dump(dm_predicted, file)
