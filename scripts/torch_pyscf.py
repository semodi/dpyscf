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
# from atoms import *
from datetime import datetime
import sys
import shutil
import os
import psutil
import gc
import tarfile

process = psutil.Process(os.getpid())
dpyscf_dir = os.environ.get('DPYSCF_DIR','..')

RHO_mult = 25
E_mult = 0.0
DEVICE = 'cpu'
HYBRID=False
keep_net_fixed=False

    
        
def get_scf(path=''):

#     x = XC_L(device=DEVICE,n_input=1, n_hidden=16, spin_scaling=True, use=[1], lob=True) # PBE_X
#     x.load_state_dict(torch.load('pbe_new/pbe_x_16_new',map_location=torch.device('cpu')))

#     c = C_L(device=DEVICE,n_input=3, n_hidden=16, use=[2])
#     c.load_state_dict(torch.load('pbe_new/pbe_c_16_new',map_location=torch.device('cpu')))
#     xc_level = 2
    
    x = XC_L(device=DEVICE,n_input=2, n_hidden=16, spin_scaling=True, use=[1,2], lob=True) # PBE_X
    x.load_state_dict(torch.load(dpyscf_dir + '/models/pretrained/scan_models/scan_x_16',map_location=torch.device('cpu')))
    
#     c = XC_L(device=DEVICE,n_input=2, n_hidden=32, spin_scaling=True, use=[1,2], lob=False) # PBE_X
    c = C_L(device=DEVICE,n_input=4, n_hidden=16, use=[2,3])
    c.load_state_dict(torch.load(dpyscf_dir + '/models/pretrained/scan_models/scan_c_16',map_location=torch.device('cpu')))
    xc_level = 3
    if HYBRID:
        try:
            a = 0.75
            b = 1
            d = 0.25
            xc = XC(grid_models=[x, c], heg_mult=True, level=xc_level, model_mult=[] )
            scf = SCF(nsteps=25, xc=xc, exx=True,alpha=0.3)

            if path:
                xc.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

            xc.add_model_mult([a,b])   
            xc.add_exx_a(d)
            xc.model_mult.requires_grad=True
            xc.exx_a.requires_grad=True
        except RuntimeError:
            a = 0.75
            b = 1
            d = 0.25
            xc = XC(grid_models=[x, c], heg_mult=True, level=xc_level, model_mult=[a, b],exx_a=d)
            scf = SCF(nsteps=25, xc=xc, exx=True,alpha=0.3)

            if path:
                xc.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            xc.model_mult.requires_grad=True
            xc.exx_a.requires_grad=True

    else:
        xc = XC(grid_models=[x, c], heg_mult=True, level=xc_level)
        scf = SCF(nsteps=25, xc=xc, exx=False,alpha=0.3)
        if path:
            xc.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    
    scf.xc.train()    
    return scf 

if __name__ == '__main__':
    if HYBRID:
        logpath = 'log/' + str(datetime.now()).replace(' ','_') 
    else:
        logpath = 'log/' + str(datetime.now()).replace(' ','_')
    def print(*args):
        with open(logpath + '.log','a') as logfile:
            logfile.write(' ,'.join([str(a) for a in args]) + '\n')
            
    try:
        os.mkdir(logpath.split('/')[0])
    except FileExistsError:
        pass
    
    with tarfile.open(logpath + 'tar.gz', "w:gz") as tar:
        source_dir = dpyscf_dir + '/dpyscf/'
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        source_dir = __file__
        tar.add(source_dir, arcname=os.path.basename(source_dir))

    
    atoms = read(dpyscf_dir + '/data/haunschild_training.traj',':')
    indices = np.arange(len(atoms)).tolist()
    pol ={'Be':True, 'HBeH':True, 'FF':False,'OCO':True,'ClCl':True, 'OO':True}

#     select = [0, 16]
#     atoms = [atoms[sel] for sel in select]
#     indices = [indices[sel] for sel in select] 

    if HYBRID:
        pop = [12, 7,  5] # (Hybrid GGA)
    else:
        pop = [21, 12, 10, 8, 7, 6, 5, 4,3, 2, 0] # (Meta-GGA)
        
    [atoms.pop(i) for i in pop]
    [indices.pop(i) for i in pop]


  
    dataset = MemDatasetRead('/gpfs/home/smdick/smdick/.data_scan/test', skip=pop)

#     dataset = MemDatasetRead('/gpfs/home/smdick/smdick/.data/test', skip= [12, 8, 5, 4])

    dataset_train = dataset

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False) # Dont change batch size !
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False) # Dont change batch size !

    molecules = {'{:3d}'.format(idx) + ''.join(a.get_chemical_symbols()): [idx] for idx, a in enumerate(atoms) if len(a.positions) > 1 }
    pure_atoms = {''.join(a.get_chemical_symbols()): [idx] for idx, a in enumerate(atoms) if len(a.positions) == 1 }

    def split(el):
            import re
            res_list = [s for s in re.split("([A-Z][^A-Z]*)", el) if s]
            return res_list

    for molecule in molecules:
        comp = []
        for a in split(molecule[3:]):
            comp.append(pure_atoms[a][0])
        molecules[molecule] += comp

    a_count = {a: np.sum([a in molecules[mol] for mol in molecules]) for a in np.unique([m  for mol in molecules for m in molecules[mol]])}


    best_loss = 1e6

    if len(sys.argv) > 1:
        scf = get_scf(sys.argv[1])
    else:
        scf = get_scf()
        scf.xc.evaluate()


#         Es = []
#         cnt = 0 
#         for dm_init, matrices, e_ref, dm_ref in dataloader_train:
#             print(atoms[cnt])
#             cnt += 1
#             dm_init = dm_init.to(DEVICE)
#             e_ref = e_ref.to(DEVICE)
#             dm_ref = dm_ref.to(DEVICE)
#             matrices = {key:matrices[key].to(DEVICE) for key in matrices}

#             results = scf.forward(matrices['dm_realinit'], matrices)
#             E = results['E']
#             Es.append(E.detach().cpu().numpy())


#         print(str(np.array(Es)[:,-1]))
#         print(str(np.array(Es)[:,-1]-np.array(Es)[:,-2]))


    scf.xc.train()
    PRINT_EVERY=1
    skip_steps = 20

    def get_optimizer(model, path=''):
        if HYBRID:
            if keep_net_fixed:
                optimizer = torch.optim.Adam([model.xc.model_mult, model.xc.exx_a],
                             lr=0.001,weight_decay=0)
            else:
                optimizer = torch.optim.Adam(list(model.parameters()) + [model.xc.model_mult, model.xc.exx_a],
                                          lr=0.0001, weight_decay=1e-8)
#         #                              lr=0.001, weight_decay=1e-8)
        else:
            optimizer = torch.optim.Adam(model.parameters(),
                                      lr=0.0000, weight_decay=1e-8)
#                                       lr=0.0001, weight_decay=1e-8)

        MIN_RATE = 1e-7
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               verbose=True, patience=int(10/PRINT_EVERY), 
                                                               factor=0.1, min_lr=MIN_RATE)

        if path:
            optimizer.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        return optimizer, scheduler

    optimizer, scheduler = get_optimizer(scf)

    AE_mult = 1

    mol_losses = {"rho" : (partial(rho_loss,loss = torch.nn.MSELoss()), RHO_mult)}
    #              "Conv" :(partial(econv_loss,loss = torch.nn.MSELoss(),skip_steps=skip_steps), 1e-6)}

    # No rho loss for atoms
    atm_losses = {}
    #                "Conv" :(partial(econv_loss,loss = torch.nn.MSELoss(),skip_steps=skip_steps), 1e-6)}

    ae_loss = partial(ae_loss,loss = torch.nn.MSELoss())

    chkpt_idx = 0
    validate_every = 10
    for epoch in range(100000):


        encountered_nan = True
        while(encountered_nan):
            error_cnt = 0
            running_losses = {key:0 for key in mol_losses}
            running_losses['ae'] = 0 
            total_loss = 0
            atm_cnt = {}
            encountered_nan = False
            try:
                for molecule in list(molecules.keys()):
                    ref_dict = {}
                    pred_dict = {}
                    loss = 0
                    for idx, data in enumerate(dataloader_train):
                        if not idx in molecules[molecule]: continue
                        print(atoms[idx])
                        dm_init, matrices, e_ref, dm_ref = data
                        dm_init = dm_init.to(DEVICE)
                        e_ref = e_ref.to(DEVICE)
                        dm_ref = dm_ref.to(DEVICE)
                        matrices = {key:matrices[key].to(DEVICE) for key in matrices}
                        dm_mix = matrices['dm_realinit']
                        mixing = torch.rand(1)*0+1
    #                     mixing = torch.rand(1)/2+0.75
                        results = scf(dm_init*(1-mixing) + dm_mix*mixing, matrices)
                        results['dm_ref'] = dm_ref
                        results['fcenter'] = matrices.get('fcenter',None)
                        results['rho'] = matrices['rho']
                        results['ao_eval'] = matrices['ao_eval']
                        results['grid_weights'] = matrices['grid_weights']
                        results['E_ref'] = e_ref
                        results['mo_energy_ref'] = matrices['mo_energy']
                        results['n_elec'] = matrices['n_elec']
                        results['e_ip_ref'] = matrices['e_ip']
                        results['mo_occ'] = matrices['mo_occ']
                        if len(atoms[idx].positions) > 1:
                            losses = mol_losses
#                         elif (not str(atoms[idx].symbols) in ['H','Be','Li']):
#                             losses = atm_losses
                        else:
                            losses = atm_losses
                        losses_eval = {key: losses[key][0](results)/a_count[idx] for key in losses}
                        running_losses.update({key:running_losses[key] +                               losses_eval[key].item() for key in losses})
                        ref_dict[''.join(atoms[idx].get_chemical_symbols())] = e_ref
                        if len(atoms[idx].positions) > 1:
                            pred_dict[''.join(atoms[idx].get_chemical_symbols())] = results['E'][skip_steps:]
                        else:
                            pred_dict[''.join(atoms[idx].get_chemical_symbols())] = results['E'][-1:]
                        loss += sum([losses_eval[key]*losses[key][1] for key in losses])

                    ael = ae_loss(ref_dict,pred_dict)
                    running_losses['ae'] += ael.item()
#                     print(molecule, ael.item())
                    loss += AE_mult * ael
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            except RuntimeError:
                encountered_nan = True
                chkpt_idx -= 1
                print('NaNs encountered, rolling back to checkpoint {}'.format(chkpt_idx%3))
                if chkpt_idx == -1:
                    scf = get_scf()
                    optimizer, scheduler = get_optimizer(scf)
                else:
                    scf = get_scf(logpath + '_{}.chkpt'.format(chkpt_idx%3))
                    optimizer, scheduler = get_optimizer(scf, logpath + '_{}.adam.chkpt'.format(chkpt_idx%3))
                scf.xc.train()
                error_cnt +=1
                if error_cnt > 3:
                    print('NaNs could not be resolved by rolling back to checkpoint')
                    raise RuntimeError('NaNs could not be resolved by rolling back to checkpoint')

        if epoch%PRINT_EVERY==0:
            running_losses = {key:np.sqrt(running_losses[key]/len(molecules))*1000 for key in running_losses}
            total_loss = np.sqrt(total_loss/len(molecules))*1000
            best_loss = min(total_loss, best_loss)
            chkpt_str = ''
            torch.save(scf.xc.state_dict(), logpath + '_current.chkpt')
            torch.save(scf, logpath + '_current.pt')
            if total_loss == best_loss:
                torch.save(scf.xc.state_dict(), logpath + '_{}.chkpt'.format(chkpt_idx%3))
                torch.save(optimizer.state_dict(), logpath + '_{}.adam.chkpt'.format(chkpt_idx%3))
                chkpt_str = '_{}.chkpt'.format(chkpt_idx%3)
                chkpt_idx += 1
            print('Epoch {} ||'.format(epoch), [' {} : {:.6f}'.format(key,val) for key, val in running_losses.items()],
                  '|| total loss {:.6f}'.format(total_loss),chkpt_str)
            if HYBRID:
                print(scf.xc.model_mult , scf.xc.exx_a)
            scheduler.step(total_loss)





