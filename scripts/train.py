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
from datetime import datetime
import sys
import shutil
import os
import psutil
import gc
import tarfile
import argparse
import json

process = psutil.Process(os.getpid())
dpyscf_dir = os.environ.get('DPYSCF_DIR','..')
DEVICE = 'cpu'

parser = argparse.ArgumentParser(description='Train xc functional')
parser.add_argument('pretrain_loc', action='store', type=str, help='Location of pretrained models (should be directory containing x and c)')
parser.add_argument('type', action='store', choices=['GGA','MGGA'])
parser.add_argument('datapath', action='store', type=str, help='Location of precomputed matrices (run prep_data first)')
parser.add_argument('--n_hidden', metavar='n_hidden', type=int, default=16, help='Number of hidden nodes (16)')
parser.add_argument('--hyb_par', metavar='hyb_par', type=float, default=0.0, help='Hybrid mixing parameter (0.0)')
parser.add_argument('--E_weight', metavar='e_weight', type=float, default=0.0, help='Weight of total energy term in loss function (0)')
parser.add_argument('--rho_weight', metavar='rho_weight', type=float, default=25, help='Weight of density term in loss function (25)')
parser.add_argument('--modelpath', metavar='modelpath', type=str, default='', help='Checkpoint location to continue training')
parser.add_argument('--logpath', metavar='logpath', type=str, default='log/', help='Logging directory (log/)')
parser.add_argument('--testrun', action='store_true', help='Do a test run over all molecules before training')
parser.add_argument('--lr', metavar='lr', type=float, default=0.0001, help='Learning rate (0.0001)')
parser.add_argument('--l2', metavar='l2', type=float, default=1e-8, help='Weight decay (1e-8)')
parser.add_argument('--hnorm', action='store_true', help='Use H energy and density in loss')
parser.add_argument('--print_stdout', action='store_true', help='Print to stdout instead of logfile')
parser.add_argument('--print_names', action='store_true', help='Print molecule names during training')


args = parser.parse_args()

RHO_mult = args.rho_weight
E_mult = args.E_weight
HYBRID = (args.hyb_par > 0.0)

def get_scf(path=args.modelpath):

    if args.type == 'GGA':
        x = XC_L(device=DEVICE,n_input=1, n_hidden=16, spin_scaling=True, use=[1], lob=True) # PBE_X
        c = C_L(device=DEVICE,n_input=3, n_hidden=16, use=[2])
        xc_level = 2
    elif args.type == 'MGGA':
        x = XC_L(device=DEVICE,n_input=2, n_hidden=16, spin_scaling=True, use=[1,2], lob=True) # PBE_X
        c = C_L(device=DEVICE,n_input=4, n_hidden=16, use=[2,3])
        xc_level = 3
    print("Loading pre-trained models from " + args.pretrain_loc)
    x.load_state_dict(torch.load(args.pretrain_loc + '/x'))
    c.load_state_dict(torch.load(args.pretrain_loc + '/c'))

    if HYBRID:
        try:
            a = 1- args.hyb_par
            b = 1
            d = args.hyb_par
            xc = XC(grid_models=[x, c], heg_mult=True, level=xc_level, model_mult=[] )
            scf = SCF(nsteps=25, xc=xc, exx=True,alpha=0.3)

            if path:
                xc.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

            xc.add_model_mult([a,b])
            xc.add_exx_a(d)
            xc.model_mult.requires_grad=True
            xc.exx_a.requires_grad=True
        except RuntimeError:
            a = 1 - args.hyb_par
            b = 1
            d = args.hyb_par
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
        logpath = args.logpath + str(datetime.now()).replace(' ','_')
    else:
        logpath = args.logpath + str(datetime.now()).replace(' ','_')

    if not args.print_stdout:
        def print(*args):
            with open(logpath + '.log','a') as logfile:
                logfile.write(' ,'.join([str(a) for a in args]) + '\n')

    try:
        os.mkdir('/'.join(logpath.split('/')[:-1]))
    except FileExistsError:
        pass

    print(json.dumps(args.__dict__,indent=4))
    with open(logpath+'.config','w') as file:
        file.write(json.dumps(args.__dict__,indent=4))

    with tarfile.open(logpath + '.tar.gz', "w:gz") as tar:
        source_dir = dpyscf_dir + '/dpyscf/'
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        source_dir = __file__
        tar.add(source_dir, arcname=os.path.basename(source_dir))

    atoms = read(dpyscf_dir + '/data/haunschild_training.traj',':')
    # atoms = read(dpyscf_dir + '/data/haunschild_test.traj',':')
    indices = np.arange(len(atoms)).tolist()

    if args.type == 'GGA':
        pop = [12, 8, 7,  5, 4]
        if HYBRID:
             pop = [12, 8, 7,  5, 4, 2] # (Hybrid GGA)
    else:
        pop = [21, 12, 11, 10, 8, 7, 5, 4, 0] # (Meta-GGA)

    # pop = []
    [atoms.pop(i) for i in pop]
    [indices.pop(i) for i in pop]



    dataset = MemDatasetRead(args.datapath, skip=pop)

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

    scf = get_scf(args.modelpath)

    if args.testrun:
        print("\n ======= Starting testrun ====== \n\n")
        scf.xc.evaluate()
        Es = []
        E_pretrained = []
        cnt = 0
        for dm_init, matrices, e_ref, dm_ref in dataloader_train:
            print(atoms[cnt])
            cnt += 1
            dm_init = dm_init.to(DEVICE)
            e_ref = e_ref.to(DEVICE)
            dm_ref = dm_ref.to(DEVICE)
            matrices = {key:matrices[key].to(DEVICE) for key in matrices}
            E_pretrained.append(matrices['e_pretrained'])
            results = scf.forward(matrices['dm_realinit'], matrices)
            E = results['E']
            Es.append(E.detach().cpu().numpy())

        e_premodel = np.array(Es)[:,-1]
        print("\n ------- Statistics ----- ")
        print(str(e_premodel), 'Energies from pretrained model' )
        print(str(np.array(E_pretrained)),'Energies from exact DFT baseline')
        print(str(e_premodel - np.array(E_pretrained)), 'Pretraining error')
        print(str(np.array(Es)[:,-1]-np.array(Es)[:,-2]), 'Convergence')

    print("\n ======= Starting training ====== \n\n")

    scf.xc.train()
    PRINT_EVERY=1
    skip_steps = 20

    def get_optimizer(model, path=''):
        if HYBRID:
            optimizer = torch.optim.Adam(list(model.parameters()) + [model.xc.model_mult, model.xc.exx_a],
                                      lr=args.lr, weight_decay=args.l2)
        else:
            optimizer = torch.optim.Adam(model.parameters(),
                                      lr=args.lr, weight_decay=args.l2)

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
    atm_losses = {}
    h_losses = {"rho" : (partial(rho_loss,loss = torch.nn.MSELoss()), RHO_mult),
                "E":  (partial(energy_loss, loss = torch.nn.MSELoss()), E_mult)}

    ae_loss = partial(ae_loss,loss = torch.nn.MSELoss())

    chkpt_idx = 0
    validate_every = 10
    for epoch in range(100000):


        encountered_nan = True
        while(encountered_nan):
            error_cnt = 0
            running_losses = {key:0 for key in mol_losses}
            running_losses['ae'] = 0
            if args.hnorm:
                running_losses['E'] = 0
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
                        if args.print_names: print(atoms[idx])
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
                        elif str(atoms[idx].symbols) in ['H'] and args.hnorm:
                            losses = h_losses
                            results['E_ref'] = -0.5
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
#                 chkpt_idx += 1
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
