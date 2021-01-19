import pyscf
from pyscf import gto,dft,scf
from pyscf.dft import radi
import torch
import numpy as np
import scipy
from ase import Atoms
from ase.io import read
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pickle
import numpy
from pyscf.dft import radi
import os




def get_mlovlp(mol, auxmol):
    """ Returns three center-one electron intergrals need for basis
    set projection
    """
    pmol = mol + auxmol
    nao = mol.nao_nr()
    naux = auxmol.nao_nr()
    eri3c = pmol.intor('int3c1e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + auxmol.nbas))

    return eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)


class Dataset(object):
    def __init__(self, **kwargs):
        needed = ['dm_init','v','t','s','eri','n_elec','e_nuc','Etot','dm']
        for n in needed:
            assert n in kwargs

        attrs = []
        for name, val in kwargs.items():
            attrs.append(name)
            setattr(self, name,val)
        self.attrs = attrs



    def __getitem__(self, index):
        mo_occ = self.mo_occ[index]
        s_oh = np.linalg.inv(np.linalg.cholesky(self.s[index]))
        s_inv_oh = s_oh.T

        matrices = {attr: getattr(self,attr)[index] for attr in self.attrs}
        dm_init = matrices.pop('dm_init')
        Etot = matrices.pop('Etot')
        dm = matrices.pop('dm')
        matrices['mo_occ'] = mo_occ
        matrices['s_inv_oh'] = s_inv_oh
        matrices['s_oh'] = s_oh
        if hasattr(self, 'ml_ovlp'):
            ml_ovlp = self.ml_ovlp[index]
            ml_ovlp = ml_ovlp.reshape(*ml_ovlp.shape[:2],self.n_atoms[index], -1)
            matrices['ml_ovlp'] = ml_ovlp
        return dm_init, matrices, Etot, dm

    def __len__(self):
        return len(self.dm_init)


class MemDatasetWrite(object):
    def __init__(self, loc='data/', **kwargs):
        self.loc = os.path.join(loc,'data')
        needed = ['dm_init','v','t','s','eri','n_elec','e_nuc','Etot','dm']
        for n in needed:
            assert n in kwargs

        keys = list(kwargs.keys())
        self.length = len(kwargs[keys[0]])
        np.save(self.loc + '_len.npy',self.length )
        for idx in range(self.length):
            datapoint = {}
            for key in keys:
                datapoint[key] = kwargs[key][idx]
            with open(self.loc + '_{}.pckl'.format(idx),'wb') as datafile:
                datafile.write(pickle.dumps(datapoint))

class MemDatasetRead(object):
    def __init__(self, loc='data/',skip=[], **kwargs):
        self.loc = os.path.join(loc,'data')
        self.length = int(np.load(self.loc + '_len.npy'))
        index_map = np.arange(self.length)
        if skip:
            index_map = np.delete(index_map, np.array(skip))
        self.index_map = index_map
        self.length = len(index_map)

    def __getitem__(self, index):
        index = self.index_map[index]
        with open(self.loc + '_{}.pckl'.format(index),'rb') as datafile:
            kwargs = pickle.loads(datafile.read())

        attrs = []
        for name, val in kwargs.items():
            attrs.append(name)
            setattr(self, name,val)
        self.attrs = attrs

        mo_occ = self.mo_occ
        s_oh = np.linalg.inv(np.linalg.cholesky(self.s))
        s_inv_oh = s_oh.T

        matrices = {attr: getattr(self,attr)for attr in self.attrs}
        dm_init = matrices.pop('dm_init')
        Etot = matrices.pop('Etot')
        dm = matrices.pop('dm')
        matrices['mo_occ'] = mo_occ
        matrices['s_inv_oh'] = s_inv_oh
        matrices['s_oh'] = s_oh
        if hasattr(self, 'ml_ovlp'):
            ml_ovlp = self.ml_ovlp
            ml_ovlp = ml_ovlp.reshape(*ml_ovlp.shape[:2],self.n_atoms, -1)
            matrices['ml_ovlp'] = ml_ovlp
        return dm_init, matrices, Etot, dm

    def __len__(self):
        return self.length

def make_rdm1_val(mo_coeff, mo_occ, n_core):

    mo_occ_val = np.array(mo_occ)
    mo_occ_val[:int(n_core/2)] = 0
    return np.einsum('ik,k,kl',mo_coeff,mo_occ_val,mo_coeff.T)

def gen_atomic_grids(mol, atom_grid={}, radi_method=radi.gauss_chebyshev,
                     level=3, nang=20, prune=dft.gen_grid.nwchem_prune, **kwargs):
    '''
    Adapted from pyscf code.
    Generate number of radial grids and angular grids for the given molecule.

    Returns:
        A dict, with the atom symbol for the dict key.  For each atom type,
        the dict value has two items: one is the meshgrid coordinates wrt the
        atom center; the second is the volume of that grid.
    '''
    if isinstance(atom_grid, (list, tuple)):
        atom_grid = dict([(mol.atom_symbol(ia), atom_grid)
                          for ia in range(mol.natm)])
    atom_grids_tab = {}
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)

        if symb not in atom_grids_tab:
            chg = gto.charge(symb)
            if symb in atom_grid:
                n_rad, n_ang = atom_grid[symb]
                if n_ang not in LEBEDEV_NGRID:
                    if n_ang in LEBEDEV_ORDER:
                        logger.warn(mol, 'n_ang %d for atom %d %s is not '
                                    'the supported Lebedev angular grids. '
                                    'Set n_ang to %d', n_ang, ia, symb,
                                    LEBEDEV_ORDER[n_ang])
                        n_ang = LEBEDEV_ORDER[n_ang]
                    else:
                        raise ValueError('Unsupported angular grids %d' % n_ang)
            else:
                n_rad = dft.gen_grid._default_rad(chg, level)
#                 n_ang = dft.gen_grid._default_ang(chg, level)
            rad, dr = radi_method(n_rad, chg, ia, **kwargs)

#             rad_weight = 4*numpy.pi * rad**2 * dr

            phi, dphi = np.polynomial.legendre.leggauss(nang)
            phi = np.arccos(phi)

            x = np.outer(rad, np.cos(phi)).flatten()
            y = np.outer(rad, np.sin(phi)).flatten()

            dxy = np.outer(dr*rad**2, dphi)

            weights = (dxy*2*np.pi).flatten()
        #     coords = np.stack([y, 0*y, x],axis=-1)
            coords = np.stack([y, 0*y, x],axis=-1)

            atom_grids_tab[symb] = coords, weights
    return atom_grids_tab

def half_circle_scan(mf, mol, level, n_ang = 25):

    atom_grids_tab = gen_atomic_grids(mol,level=level,nang=n_ang)

    coords, weights = dft.gen_grid.gen_partition(mol, atom_grids_tab,radi.treutler_atomic_radii_adjust, atomic_radii=radi.BRAGG_RADII, becke_scheme=dft.gen_grid.stratmann)

    g = dft.gen_grid.Grids(mol)
    g.coords = coords
    g.weights = weights

    dm = mf.make_rdm1()
    if dm.ndim ==3:
        dm = np.sum(dm,axis=0)
    pruned = dft.rks.prune_small_rho_grids_(mf,mol,dm, g)
#     pruned = g
    print('Number of grid points (level = {}, n_ang = {}):'.format(level,n_ang), len(pruned.weights))
    coords = pruned.coords
    weights = pruned.weights
#     coords2 = np.array(coords)
#     coords2[:,0] *= -1
#     coords = np.concatenate([coords,coords2])
#     weights = np.concatenate([weights, weights])/2

    coords2 = np.array(coords)
    coords2 = coords2[:,[1,0,2]]

    s = 1/np.sqrt(2)
    coords3 = np.array(coords)
    coords3 = coords3.dot(np.array([[s,s,0],[-s,s,0],[0,0,1]]))
    coords = np.concatenate([coords,coords2, coords3])
    weights = np.concatenate([weights, weights, weights])/3
    return coords, weights

def half_circle(n_rad = 100, n_ang = 70):
    r, dr = radi.treutler_ahlrichs(n_rad)

    phi, dphi = np.polynomial.legendre.leggauss(n_ang)
    phi = np.arccos(phi)

    x = np.outer(r, np.cos(phi)).flatten()
    y = np.outer(r, np.sin(phi)).flatten()

    dxy = np.outer(dr*r**2, dphi)

    weights = (dxy*2*np.pi).flatten()
#     coords = np.stack([y, 0*y, x],axis=-1)
    coords = np.stack([y, 0*y, x],axis=-1)

    return coords, weights

def line(n_rad = 100, n_ang = 70):
    r, dr = radi.treutler_ahlrichs(n_rad)


    dxy = dr*r**2

    weights = (dxy*4*np.pi).flatten()
#     coords = np.stack([y, 0*y, x],axis=-1)
    coords = np.stack([r*0, 0*r, r],axis=-1)

    return coords, weights

def get_datapoint(atoms, xc='', basis='6-311G*', ncore=0, grid_level=0,
                  nl_cutoff=0, grid_deriv=1, init_guess = False, ml_basis='',
                  do_fcenter=True, zsym = False, n_rad=20,n_ang=10, spin=0, pol=False,
                  ref_basis='', ref_path = '', ref_index=0):

    print(atoms)
    print(basis)
    if not ref_basis:
        ref_basis = basis

    if xc:
        if spin == 0 and not pol:
            method = dft.RKS
        else:
            method = dft.UKS
    else:
#         method = scf.RHF
        method = scf.UHF

    features = {}

    pos = atoms.positions
    spec = atoms.get_chemical_symbols()
    mol_input = [[s, p] for s, p in zip(spec, pos)]
    mol = gto.M(atom=mol_input, basis=basis,spin=spin)
    mol_ref = gto.M(atom=mol_input, basis=ref_basis, spin=spin)

    if ml_basis:
        auxmol = gto.M(atom=mol_input,spin=spin, basis=gto.parse(open(ml_basis,'r').read()))
        ml_ovlp = get_mlovlp(mol,auxmol)
        features.update({'ml_ovlp':ml_ovlp})
    s = mol.intor('int1e_ovlp')
    t = mol.intor('int1e_kin')
    v = mol.intor('int1e_nuc')
    eri = mol.intor('int2e')
    if do_fcenter:
        fcenter = mol.intor('int4c1e')


    mf = method(mol)
    if xc:
        mf.xc = xc
        if grid_level:
            mf.grids.level = grid_level
        else:
            mf.grids.level = 6

    mf.kernel()

#         dm_base = make_rdm1_val(mf.mo_coeff,mf.mo_occ, 0)
    dm_base = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    e_base = mf.energy_tot()
    e_ip = mf.mo_energy[...,:-1][np.diff(mf.mo_occ).astype(bool)]
    if len(e_ip) == 2:
        e_ip = e_ip[0]
        ip_idx = np.where(np.diff(mf.mo_occ).astype(bool))[1][0]
    else:
        ip_idx = np.where(np.diff(mf.mo_occ).astype(bool))[0]


    if init_guess:
        dm_init = mf.get_init_guess()
    else:
        dm_init = np.array(dm_base)
        dm_realinit = mf.get_init_guess()

    n_elec = np.array([sum(atoms.get_atomic_numbers())])
    n_atoms = np.array(len(atoms)).astype(int)


    matrices = {'dm_init':dm_init,
                'v':v,
                't':t,
                's':s,
                'eri':eri,
                'n_elec':n_elec,
                'n_atoms':n_atoms,
                'e_nuc': np.array(mf.energy_nuc()),
                'mo_energy': np.array(mf.mo_energy),
                'mo_occ': np.array(mf.mo_occ),
                'e_ip' : e_ip,
                'ip_idx': ip_idx,
                'e_pretrained': e_base}


    if not init_guess:
        matrices['dm_realinit'] = dm_realinit

    if ref_path:
        dm_base = np.load(ref_path+ '/{}.dm.npy'.format(ref_index))
        if method == dft.UKS and dm_base.ndim == 2:
            dm_base = np.stack([dm_base,dm_base])*0.5
        if method == dft.RKS and dm_base.ndim == 3:
            dm_base = np.sum(dm_base, axis=0)
        e_base =  (pd.read_csv(ref_path + '/energies', delim_whitespace=True,header=None,index_col=0).loc[ref_index]).values.flatten()[0]


    if grid_level:
        if spin ==0 and not pol:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        mf.grids.level = grid_level
        if xc:
            mf.xc = xc
        mf.kernel()
        if zsym and not nl_cutoff:
            if len(pos) == 1:
                method = line
#                 method = half_circle
            else:
                method = half_circle
            mf.grids.coords, mf.grids.weights, L, scaling = get_symmetrized_grid(mol, mf, n_rad, n_ang, method=method)
            features.update({'L': L, 'scaling': scaling})
        else:
            features.update({'L': np.eye(dm_init.shape[-1]), 'scaling': np.ones([dm_init.shape[-1]]*2)})
        if spin != 0 or pol:
            rho_a = get_rho(mf, mol_ref, dm_base[0], mf.grids)
            rho_b = get_rho(mf, mol_ref, dm_base[1], mf.grids)
            rho = np.stack([rho_a,rho_b],axis=0)
        else:
            rho = get_rho(mf, mol_ref, dm_base, mf.grids)

        features.update({'rho': rho,
                         'ao_eval':mf._numint.eval_ao(mol, mf.grids.coords, deriv=grid_deriv),
                         'grid_weights':mf.grids.weights})
#         if nl_cutoff:
#             grid_coords = mf.grids.coords
#             coords = grid_coords
#             X1, X2 = [x.flatten() for x in np.meshgrid(*[np.arange(len(coords))]*2)]
#             dr = np.linalg.norm(coords[X1] - coords[X2],axis=-1)
#             dist_filt = dr < nl_cutoff
#             edge_index = np.concatenate([X1.reshape(-1,1)[dist_filt],X2.reshape(-1,1)[dist_filt]],axis=-1).T
#             features.update({'grid_coords': grid_coords, 'edge_index': edge_index})

    if do_fcenter:
        features.update({'fcenter':fcenter})

    matrices.update(features)

    return e_base, np.eye(3), matrices


def get_rho(mf, mol, dm, grids):

    ao_eval = mf._numint.eval_ao(mol, grids.coords)
    rho = mf._numint.eval_rho(mol, ao_eval, dm)
    return rho

def get_L(mol, method):


    if method==half_circle:
        pys = np.where(['py' in l for l in mol.ao_labels()])[0]
        pxs = np.where(['px' in l for l in mol.ao_labels()])[0]

        dxys = np.where(['dxy' in l for l in mol.ao_labels()])[0]
        dx2y2s = np.where(['dx2-y2' in l for l in mol.ao_labels()])[0]
        dyzs = np.where(['dyz' in l for l in mol.ao_labels()])[0]

        fx3s = np.where(['fx^3' in l for l in mol.ao_labels()])[0]
        fy3s = np.where(['fy^3' in l for l in mol.ao_labels()])[0]

        fzx2s = np.where(['fzx^2' in l for l in mol.ao_labels()])[0]
        fxyzs =  np.where(['fxyz' in l for l in mol.ao_labels()])[0]

        fxz2s = np.where(['fxz^2' in l for l in mol.ao_labels()])[0]
        fyz2s =  np.where(['fyz^2' in l for l in mol.ao_labels()])[0]


        L = np.eye(len(mol.ao_labels()))

        for px,py in zip(pxs,pys):
            L[py,px] = 1
        for dxy,dx2y2,dyz in zip(dxys,dx2y2s,dyzs):
            L[dxy,dx2y2] = 1
            L[dyz,dx2y2] = 1

        for fx3, fy3 in zip(fx3s,fy3s):
            L[fy3,fx3] = 1

        for fzx2, fxyz in zip(fzx2s,fxyzs):
            L[fxyz,fzx2] = 1

        for fxz2, fyz2 in zip(fxz2s,fyz2s):
            L[fyz2, fxz2] = 1

    else:
        ps = np.where(['p' in l and not 'z' in l for l in mol.ao_labels()])[0]
        pzs = np.where(['pz' in l for l in mol.ao_labels()])[0]
        if len(pzs):
             ps = ps.reshape(len(pzs), -1)

        ds = np.where(['d' in l and not 'z^2' in l for l in mol.ao_labels()])[0]
        dzs = np.where(['dz^2' in l for l in mol.ao_labels()])[0]
        if len(dzs):
            ds = ds.reshape(len(dzs), -1)

        fs = np.where(['f' in l and not 'z^3' in l for l in mol.ao_labels()])[0]
        fzs = np.where(['fz^3' in l for l in mol.ao_labels()])[0]
        if len(fzs):
            fs = fs.reshape(len(fzs), -1)

        L = np.eye(len(mol.ao_labels()))

        for p, pz in zip(ps, pzs):
            for p_ in p:
                L[p_,pz] = 1

        for d, dz in zip(ds, dzs):
            for d_ in d:
                L[d_,dz] = 1
#                 L[dz,d_] = 1

        for f, fz in zip(fs, fzs):
            for f_ in f:
                L[f_,fz] = 1

    return L

def get_symmetrized_grid(mol, mf, n_rad=20, n_ang=10, print_stat=True, method= half_circle, return_errors = False):
    dm = mf.make_rdm1()
    if dm.ndim != 3:
#         dm = np.sum(dm, axis=0)
        dm = np.stack([dm,dm],axis=0)*0.5
#         dm = dm[0]
#     rho_ex = mf._numint.get_rho(mol, dm, mf.grids)
    rho_ex_a = mf._numint.eval_rho(mol, mf._numint.eval_ao(mol, mf.grids.coords, deriv=2) , dm[0], xctype='metaGGA')
    rho_ex_b = mf._numint.eval_rho(mol, mf._numint.eval_ao(mol, mf.grids.coords, deriv=2) , dm[1], xctype='metaGGA')

    q_ex_a = np.sum(rho_ex_a[0] * mf.grids.weights)
    q_ex_b = np.sum(rho_ex_b[0] * mf.grids.weights)

    exc_ex = np.sum(mf._numint.eval_xc(mf.xc, (rho_ex_a,rho_ex_b),spin=1)[0]*mf.grids.weights*(rho_ex_a[0]+rho_ex_b[0]))

    print("Using method", method, " for grid symmetrization")
    if mf.xc == 'SCAN' and method == half_circle:
        meta = True
        coords, weights = half_circle_scan(mf, mol, n_rad, n_ang)
    else:
        meta = False
        coords, weights = half_circle(n_rad, n_ang)
        atomic_grids = {mol.atom_pure_symbol(el): ( np.array(coords), np.array(weights)) for el, _ in enumerate(mol.atom_charges())}

        coords, weights = \
                mf.grids.gen_partition(mol, atomic_grids, radii_adjust=mf.grids.radii_adjust,
                                       atomic_radii=mf.grids.atomic_radii)

    exc = mf._numint.eval_xc(mf.xc, (rho_ex_a,rho_ex_b),spin=1)[0]
    vxc = mf._numint.eval_xc(mf.xc, rho_ex_a +rho_ex_b)[1][0]
    if meta:
        vtau = mf._numint.eval_xc(mf.xc, rho_ex_a +rho_ex_b)[1][3]
    aoi = mf._numint.eval_ao(mol, mf.grids.coords, deriv = 2)

    vmunu1 = np.einsum('i,i,ij,ik->jk', mf.grids.weights, vxc,aoi[0],aoi[0])
    if meta:
        vtmunu1  = np.einsum('i,lij,lik->jk',vtau*mf.grids.weights, aoi[1:4],aoi[1:4])
    emunu1 = np.einsum('i,i,ij,ik->jk',mf.grids.weights, exc,aoi[0],aoi[0])



    mf.grids.coords = coords
    mf.grids.weights = weights

    rho_sym_a = mf._numint.eval_rho(mol, mf._numint.eval_ao(mol, mf.grids.coords, deriv=2) , dm[0], xctype='metaGGA')
    rho_sym_b = mf._numint.eval_rho(mol, mf._numint.eval_ao(mol, mf.grids.coords, deriv=2) , dm[1], xctype='metaGGA')

    q_sym_a = np.sum(rho_sym_a[0] * mf.grids.weights)
    q_sym_b = np.sum(rho_sym_b[0] * mf.grids.weights)

    exc_sym = np.sum(mf._numint.eval_xc(mf.xc, (rho_sym_a,rho_sym_b),spin=1)[0]*mf.grids.weights*(rho_sym_a[0]+rho_sym_b[0]))
    if print_stat:
        print('{:10.6f}e   ||{:10.6f}e   ||{:10.6f}e'.format(q_ex_a, q_sym_a, np.abs(q_ex_a-q_sym_a)))
        print('{:10.6f}e   ||{:10.6f}e   ||{:10.6f}e'.format(q_ex_b, q_sym_b, np.abs(q_ex_b-q_sym_b)))
        print('{:10.3f}mH  ||{:10.3f}mH  ||{:10.3f}  microH'.format(1000*exc_ex, 1000*exc_sym, 1e6*np.abs(exc_ex-exc_sym)))
    error = 1e6*np.abs(exc_ex-exc_sym)

    exc = mf._numint.eval_xc(mf.xc, (rho_sym_a,rho_sym_b),spin=1)[0]
    vxc = mf._numint.eval_xc(mf.xc, rho_sym_a +rho_sym_b)[1][0]
    if meta:
        vtau = mf._numint.eval_xc(mf.xc, rho_sym_a +rho_sym_b)[1][3]
    aoi = mf._numint.eval_ao(mol, mf.grids.coords, deriv =2)

    vmunu2 = np.einsum('i,i,ij,ik->jk',mf.grids.weights, vxc,aoi[0],aoi[0])
    if meta:
        vtmunu2  = np.einsum('i,lij,lik->jk',vtau*mf.grids.weights,aoi[1:4],aoi[1:4])
    emunu2 = np.einsum('i,i,ij,ik->jk',mf.grids.weights, exc,aoi[0],aoi[0])

    if meta:
        vmunu2_sym = vmunu2
        emunu2_sym = emunu2
        L = np.eye(len(vmunu2))
    else:
        L = get_L(mol, method)
        vmunu2_sym = L.dot(vmunu2.dot(L.T))
        emunu2_sym = L.dot(emunu2.dot(L.T))

    scaling = emunu1/(emunu2_sym + 1e-10)
    vmunu2 = vmunu2_sym*scaling

    if meta:
        scalingt = vtmunu1/(vtmunu2 + 1e-10)
        vtmunu2 = vtmunu2*scalingt

    if print_stat:print({True: 'Potentials identical', False: 'Potentials not identical'}[np.allclose(vmunu1,vmunu2,atol=1e-5)])
    if print_stat and meta:print({True: 'tau Potentials identical', False: 'tau Potentials not identical'}[np.allclose(vtmunu1,vtmunu2,atol=1e-5)])

    if meta:
        return mf.grids.coords, mf.grids.weights, L, np.stack([scaling,scalingt])
    else:
        return mf.grids.coords, mf.grids.weights, L, scaling
