#!/usr/bin/env python
# coding: utf-8
import numpy as np
from ase import Atoms
from ase.io import read
from dpyscf.net import *
from dpyscf.utils import *
from ase.units import Bohr
import sys
import os

if __name__ == '__main__':

    if len(sys.argv) < 3:
        raise Exception("Must provide dataset location and functional")
    loc = sys.argv[1]
    func = sys.argv[2]
    if func not in ['PBE','SCAN']:
        raise Exception("Functional has to be either SCAN or PBE")

    if func == 'SCAN':
        atoms = read('../data/haunschild_scan.traj',':')
    elif func == 'PBE':
        atoms = read('../data/haunschild_training.traj',':')
    indices = np.arange(len(atoms))

    #TESTING
    # indices = [0, 16]
    # atoms = [atoms[i] for i in indices]
    
    # indices = [0, 11, 16, 17]
    # pol ={'Be':True, 'HBeH':True, 'FF':True,'OCO':True,'ClCl':True,'OC':True}
    pol ={}

    basis = '6-311++G(3df,2pd)'

    distances = np.arange(len(atoms))


    baseline = [get_datapoint(d, basis=d.info.get('basis',basis), grid_level=d.info.get('grid_level', 9),
                              xc=func, zsym=d.info.get('sym',True),
                              n_rad=d.info.get('n_rad',30), n_ang=d.info.get('n_ang',15),
                              init_guess=False, spin = d.info.get('spin',0),
                              pol=pol.get(''.join(d.get_chemical_symbols()), False), do_fcenter=False,
                              ref_path='../data/ref/6-311/', ref_index= idx,ref_basis='6-311++G(3df,2pd)') for idx, d in zip(indices, atoms)]

    E_base =  [r[0] for r in baseline]
    DM_base = [r[1] for r in baseline]
    inputs = [r[2] for r in baseline]
    inputs = {key: [i.get(key,None) for i in inputs] for key in inputs[0]}

    DM_ref = DM_base
    E_ref = E_base

    try:
        os.mkdir(loc)
    except FileExistsError:
        pass

    dataset = MemDatasetWrite(loc = loc, Etot = E_ref, dm = DM_ref, **inputs)
