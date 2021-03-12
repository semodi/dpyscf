import pylibnxc
from pyscf import gto, scf, dft
import json
from ase.io import read, write
import numpy as np
import sys
from ase.units import Hartree, kcal, mol
kcalpmol = kcal/mol



def test_bh(xc, nxc='nxc', indices= [], ref_dir='../'):
    with open(ref_dir+'../data/bh_76_instructions.json','r') as file:
        instructions = json.load(file)
    with open(ref_dir +'../data/bh_76_dict.json','r') as file:
        xyz_dict = json.load(file)

    basis = '6-311++G(3df,2pd)'
    barrier_heights = []
    reference_heights = []
    if indices:
        instructions = [instructions[idx] for idx in indices]
    for inst in instructions[:]:
        geometries = inst['setup']['reaction_geometries']
        xyz_data = {}
        for geom in geometries:
            xyz_data[geom] = read(ref_dir + '../data/xyz/' + xyz_dict[geom[2:]],':')


        reference_heights.append(inst['reference_value'])

        energies = []
        for system in xyz_data:

            a = xyz_data[system][0]
            spin = (a.info.get('multiplicity',1)-1)
            charge = a.info.get('charge',0)

            pos = a.positions

            this_basis = basis
            spec = a.get_chemical_symbols()
            mol_input = [[s, p] for s, p in zip(spec, pos)]
            try:
                mol = gto.M(atom=mol_input, basis=this_basis,spin=spin, charge=charge)
            except Exception:
                spin =1
                mol = gto.M(atom=mol_input, basis=this_basis,spin=spin, charge=charge)


            method = pylibnxc.pyscf.UKS


            mol.verbose=3

            if nxc  == 'nxc':
                mf = method(mol, nxc=xc, nxc_kind='grid')
            elif nxc == 'xc':
                mf = method(mol)
                mf.xc = xc

            mf.grids.level=9
            mf.kernel()
            energies.append(mf.e_tot)
        barrier_heights.append(Hartree/kcalpmol*(energies[-1] - np.sum(energies[:-1])))
    return barrier_heights, reference_heights

if __name__ == '__main__':
    barrier_heights, reference_heights = test_bh(*sys.argv[1:])
    np.save('{}_bh76.npy'.format(sys.argv[1]), barrier_heights)
    np.save('bh76.npy'.format(sys.argv[1]), reference_heights)
