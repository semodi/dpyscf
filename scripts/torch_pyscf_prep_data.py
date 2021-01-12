#!/usr/bin/env python
# coding: utf-8
import pyscf
from pyscf import gto,dft,scf
import torch
torch.set_default_dtype(torch.double)
import pyscf
from pyscf import gto,dft,scf

import numpy as np
import scipy
from ase import Atoms
from ase.io import read
from dpyscf.net import * 
from dpyscf.torch_routines import * 
from dpyscf.utils import *
from dpyscf.losses import *
from pyscf.cc import CCSD
from functools import partial
from ase.units import Bohr
from atoms import *
from datetime import datetime
import sys
import shutil
import os
import psutil
import gc
process = psutil.Process(os.getpid())





DEVICE = 'cpu'

if __name__ == '__main__':

#     def print(*args):
#         with open(logpath + '.log','a') as logfile:
#             logfile.write(' ,'.join([str(a) for a in args]) + '\n')
    
    atoms = read('../data/haunschild_scan.traj',':')
    indices = np.arange(len(atoms)).tolist()
    pol ={'Be':True, 'HBeH':True, 'FF':True,'OCO':True,'ClCl':True,'OC':True}

#     select = [-1]
#     atoms = [atoms[sel] for sel in select]
#     indices = [indices[sel] for sel in select] 
#     atoms.pop(7) # O2
#     indices.pop(7)

    basis = '6-311++G(3df,2pd)'
#     atoms[2].info['basis'] = '6-311+G*' #LiF

    distances = np.arange(len(atoms))

    # baseline = [get_datapoint(d, basis=d.info.get('basis',basis), grid_level=d.info.get('grid_level', 7),
    #                           xc='PBE',zsym=d.info.get('sym',True),
    #                           n_rad=d.info.get('n_rad',30), n_ang=d.info.get('n_ang',15), 
    #                           init_guess=False, spin = d.info.get('spin',0), 
    #                           pol=pol.get(''.join(d.get_chemical_symbols()), False), do_fcenter=False,
    #                           ref_path='ref/aug-ccpvqz/', ref_index= idx,ref_basis='aug-ccpvqz') for idx, d in zip(indices, atoms)]

    baseline = [get_datapoint(d, basis=d.info.get('basis',basis), grid_level=d.info.get('grid_level', 9),
                              xc='SCAN',zsym=d.info.get('sym',True),
                              n_rad=d.info.get('n_rad',30), n_ang=d.info.get('n_ang',15), 
                              init_guess=False, spin = d.info.get('spin',0), 
                              pol=pol.get(''.join(d.get_chemical_symbols()), False), do_fcenter=False,
                              ref_path='ref_2/6-311/', ref_index= idx,ref_basis='6-311++G(3df,2pd)') for idx, d in zip(indices, atoms)]

    E_base =  [r[0] for r in baseline]
    DM_base = [r[1] for r in baseline]
    inputs = [r[2] for r in baseline]
    inputs = {key: [i.get(key,None) for i in inputs] for key in inputs[0]}

    DM_ref = DM_base
    E_ref = E_base

    dataset = MemDatasetWrite(loc = '/gpfs/home/smdick/smdick/.data_scan/test', Etot = E_ref, dm = DM_ref, **inputs)

    


