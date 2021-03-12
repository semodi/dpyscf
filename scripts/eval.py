import numpy as np
from ase.io import read
from ase.units import Hartree,kcal,mol
kcalpmol = kcal/mol
import sys

ref = np.array([a.get_potential_energy()/Hartree for a in read('../data/haunschild_g2/g2_97.traj',':')])
pred = -np.load(sys.argv[1] + '_g2_ae.npy')

skip = [15, 58, 83, 2, 25, 113, 18]
filt = np.arange(len(ref))
filt = np.delete(filt, skip)
# H2O, NH3, NO, H2, LiF, N2, CNH

dev = ref-pred 
np.save(sys.argv[1] + '_resid_ae.npy', dev )

dev = dev[filt]
mae = np.mean(np.abs(dev))
std = np.sqrt(np.mean(dev**2))
max_ = np.max(np.abs(dev))
print(mae, 'Hartree')
print(mae*Hartree/kcalpmol, 'kcal/mol')

print(std, 'Hartree')
print(std*Hartree/kcalpmol, 'kcal/mol')

print(max_, 'Hartree')
print(max_*Hartree/kcalpmol, 'kcal/mol')


##=================== Electron density 


from pyscf import gto, scf, dft
import pickle
from dpyscf.utils import get_rho


atoms = read('../data/haunschild_g2/g2_97.traj',':')
with open(sys.argv[1] + '_g2.dm','rb') as file:
    dms_pred = pickle.load(file)
errors = []

progress_it = len(atoms)//20 
for a_idx, a in enumerate(atoms):
    if a_idx%progress_it==0:
        print('Progress: {}%'.format(a_idx//progress_it*5))
    basis = '6-311++G(3df,2pd)'
    spin = a.info.get('spin', 0)
    pos = a.positions
    this_basis = basis
    spec = a.get_chemical_symbols()
    mol_input = [[s, p] for s, p in zip(spec, pos)]
    try:
        mol = gto.M(atom=mol_input, basis=this_basis,spin=spin)
    except Exception:
        spin =1
        mol = gto.M(atom=mol_input, basis=this_basis,spin=spin)

        
    mf = dft.UKS(mol)
    mf.xc = 'PBE'
    mf.grids.level = 9
    mf.grids.build()

    dm = dms_pred[a_idx]
    dm_ref = np.load('../data/ccsdt/{}.dm.npy'.format(a_idx))
    
#     for s in [0,1]:
    rho_ref = get_rho(mf,mol,np.sum(dm_ref, axis=0), mf.grids)
    rho = get_rho(mf,mol,np.sum(dm,axis=0), mf.grids)
    errors.append(np.sqrt(np.sum((rho-rho_ref)**2*mf.grids.weights))/(np.sum(rho_ref*mf.grids.weights)))

errors = np.array(errors)
np.save(sys.argv[1] + '_errors_rho.npy', errors)

print(np.mean(errors[filt]))