import torch
from torch.nn import Sequential as Seq, Linear, ReLU,Sigmoid,GELU
# from torch_geometric.nn import MessagePassing
import pyscf
from pyscf import gto,dft,scf
import torch
import numpy as np
import scipy
from ase import Atoms
from ase.io import read
from .torch_routines import *

def get_scf(xctype, pretrain_loc, hyb_par=0, path='', DEVICE='cpu'):

    if xctype == 'GGA':
        x = XC_L(device=DEVICE,n_input=1, n_hidden=16, spin_scaling=True, use=[1], lob=1.804) # PBE_X
        c = C_L(device=DEVICE,n_input=3, n_hidden=16, use=[2], ueg_limit=True)
        xc_level = 2
    elif xctype == 'MGGA':
        x = XC_L(device=DEVICE,n_input=2, n_hidden=16, spin_scaling=True, use=[1,2], lob=1.174) # PBE_X
        c = C_L(device=DEVICE,n_input=4, n_hidden=16, use=[2,3])
        xc_level = 3
    print("Loading pre-trained models from " + pretrain_loc)
    x.load_state_dict(torch.load(pretrain_loc + '/x'))
    c.load_state_dict(torch.load(pretrain_loc + '/c'))

    if hyb_par:
        try:
            a = 1 - hyb_par
            b = 1
            d = hyb_par
            xc = XC(grid_models=[x, c], heg_mult=True, level=xc_level )
            scf = SCF(nsteps=25, xc=xc, exx=True,alpha=0.3)

            xc.add_exx_a(d)
            xc.exx_a.requires_grad=True

            if path:
                xc.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

        except RuntimeError:
            a = 1 - hyb_par
            b = 1
            d = hyb_par
            xc = XC(grid_models=[x, c], heg_mult=True, level=xc_level, exx_a=d)
            scf = SCF(nsteps=25, xc=xc, exx=True,alpha=0.3)
            print(xc.exx_a)
            if path:
                xc.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            xc.exx_a.requires_grad=True
            print(xc.exx_a)
    else:
        xc = XC(grid_models=[x, c], heg_mult=True, level=xc_level)
        scf = SCF(nsteps=25, xc=xc, exx=False,alpha=0.3)
        if path:
            xc.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    scf.xc.train()
    return scf

epsilon=0
class Reduce(torch.nn.Module):
    def __init__(self,mean, std, device='cpu'):
        super().__init__()
        self.mean = torch.Tensor([mean],).double().to(device)
        self.std = torch.Tensor([std]).double().to(device)
    def forward(self, input):
        return torch.sum(input)*self.std[0] + self.mean[0]

class Symmetrizer(torch.nn.Module):
    def __init__(self, basis, device= 'cpu'):
        super().__init__()
        auxmol = gto.M(atom='He 0 0 0', basis=gto.parse(open(basis,'r').read()))
        sli = auxmol.aoslice_by_atom()
        M = np.zeros([sli[0,1],sli[0,3]])
        idx = 0
        for shell, _ in enumerate(M):
            length = auxmol.bas_len_cart(shell)
            M[shell,idx:idx+length] = 1
            idx += length
        self.M = torch.from_numpy(M).to(device)

    def forward(self, input):
        return torch.sqrt(torch.einsum('ij,...j',self.M,torch.pow(input,2)))

def get_M(basis):
    bp = neuralxc.pyscf.BasisPadder(gto.M(atom='O 0 0 0', basis=gto.parse(open(basis,'r').read())))
    il = bp.indexing_l['O'][0]
    ir = bp.indexing_r['O'][0]
    M = np.zeros([len(il),len(ir)])
    M[il, ir] = 1
    return M


class XC(torch.nn.Module):

    def __init__(self, grid_models=None, nxc_models=None, heg_mult=True, pw_mult=True,
                    level = 1, exx_a=None, polynomial=False):
        super().__init__()
        self.polynomial = polynomial
        self.grid_models = None
        self.nxc_models = None
        self.heg_mult = heg_mult
        self.pw_mult = pw_mult
        self.edge_index = None
        self.grid_coords = None
        self.training = True
        self.level = level
        self.epsilon = 1e-7
        self.loge = 1e-8
#         self.loge = 1e-3
        self.s_gam = 1

        if heg_mult:
            self.heg_model = LDA_X()
        if pw_mult:
            self.pw_model = PW_C()
        if grid_models:
            if not isinstance(grid_models, list): grid_models = [grid_models]
            self.grid_models = torch.nn.ModuleList(grid_models)
        if nxc_models:
            if not isinstance(nxc_models, list): grid_models = [nxc_models]
            self.nxc_models = torch.nn.ModuleList(nxc_models)
#         if model_mult:
#             assert len(model_mult) == len(grid_models)
#             self.register_buffer('model_mult',torch.Tensor(model_mult))
#         else:
        self.model_mult = [1 for m in self.grid_models]
        if exx_a is not None:
            self.register_buffer('exx_a', torch.Tensor([exx_a]))
        else:
#             self.register_buffer('exx_a', torch.Tensor([0]))
            self.exx_a = 0
    def evaluate(self):
        self.training=False
    def train(self):
        self.training=True

    def add_model_mult(self, model_mult):
        del(self.model_mult)
        self.register_buffer('model_mult',torch.Tensor(model_mult))

    def add_exx_a(self, exx_a):
        self.exx_a = torch.Tensor([exx_a])


    def get_descriptors_pol(self, rho0_a, rho0_b, gamma_a, gamma_b, gamma_ab, tau_a, tau_b, spin_scaling = False):


        uniform_factor = (3/10)*(3*np.pi**2)**(2/3)

        def l_1(rho):
            return rho**(1/3)

        def l_2(rho, gamma):
            return torch.sqrt(gamma)/(2*(3*np.pi**2)**(1/3)*rho**(4/3)+self.epsilon)

        def l_3(rho, gamma, tau):
            return (tau - gamma/(8*(rho+self.epsilon)))/(uniform_factor*rho**(5/3)+self.epsilon)

        if not spin_scaling:
            zeta = (rho0_a - rho0_b)/(rho0_a + rho0_b + self.epsilon)
            spinscale = 0.5*((1+zeta)**(4/3) + (1-zeta)**(4/3)) # zeta

        if self.level > 0:
            if spin_scaling:
                # descr1 = torch.log((2*rho0_a)**(1/3) + self.loge)
                descr1 = l_1(2*rho0_a)
                # descr2 = torch.log((2*rho0_b)**(1/3) + self.loge)
                descr2 = l_1(2*rho0_b)
            else:
                # descr1 = torch.log((rho0_a + rho0_b)**(1/3) + self.loge)# rho
                descr1 = l_1(rho0_a)# rho
                descr2 = l_1(rho0_b)# rho
            descr = torch.cat([descr1.unsqueeze(-1), descr2.unsqueeze(-1)],dim=-1)
        if self.level > 1:
            if spin_scaling:
                # descr3a = torch.sqrt(4*gamma_a)/(2*(3*np.pi**2)**(1/3)*(2*rho0_a)**(4/3)+self.epsilon) # s
                descr3a = l_2(2*rho0_a, 4*gamma_a) # s
                # descr3b = torch.sqrt(4*gamma_b)/(2*(3*np.pi**2)**(1/3)*(2*rho0_b)**(4/3) +self.epsilon) # s
                descr3b = l_2(2*rho0_b, 4*gamma_b) # s
                descr3 = torch.cat([descr3a.unsqueeze(-1), descr3b.unsqueeze(-1)],dim=-1)
                # descr3 = (1-torch.exp(-descr3**2/self.s_gam))*torch.log(descr3 + 1)
            else:
                # descr3 = torch.sqrt(gamma_a + gamma_b + 2*gamma_ab)/(2*(3*np.pi**2)**(1/3)*(rho0_a + rho0_b)**(4/3)+self.epsilon) # s
                descr3a = l_2(rho0_a, gamma_a) # s
                # descr3b = torch.sqrt(4*gamma_b)/(2*(3*np.pi**2)**(1/3)*(2*rho0_b)**(4/3) +self.epsilon) # s
                descr3b = l_2(rho0_b, gamma_b) # s
                descr3 = torch.cat([descr3a.unsqueeze(-1), descr3b.unsqueeze(-1)],dim=-1)
            descr = torch.cat([descr, descr3],dim=-1)
        if self.level == 3:
            assert False
            if spin_scaling:
                # descr4a = (2*tau_a - 4*gamma_a/(16*(rho0_a+self.epsilon)))/(uniform_factor*(2*rho0_a)**(5/3)+self.epsilon)
                descr4a = l_3(2*rho0_a, 4*gamma_a, 2*tau_a)
                # descr4b = (2*tau_b - 4*gamma_b/(16*(rho0_b+self.epsilon)))/(uniform_factor*(2*rho0_b)**(5/3)+self.epsilon)
                descr4b = l_3(2*rho0_b, 4*gamma_b, 2*tau_b)
                descr4 = torch.cat([descr4a.unsqueeze(-1), descr4b.unsqueeze(-1)],dim=-1)
                descr4 = descr4**3/(descr4**2+self.epsilon)
            else:
                # descr4 = (tau_a + tau_b - (gamma_a + gamma_b+2*gamma_ab)/(8*(rho0_a + rho0_b + self.epsilon)))/(self.epsilon+(rho0_a + rho0_b)**(5/3)*((1+zeta)**(5/3) + (1-zeta)**(5/3))) # tau
                descr4 = l_3(rho0_a + rho0_b, gamma_a + gamma_b + 2*gamma_ab, tau_a + tau_b)
                descr4 = descr4/((1+zeta)**(5/3) + (1-zeta)**(5/3))
                descr4 = descr4**3/(descr4**2+self.epsilon)
                descr4 = descr4.unsqueeze(-1)
            descr4 = torch.log((descr4 + 1)/2)
            descr = torch.cat([descr, descr4],dim=-1)
        if spin_scaling:
            descr = descr.view(descr.size()[0],-1,2).permute(2,0,1)

        return descr

    def get_descriptors(self, rho0_a, rho0_b, gamma_a, gamma_b, gamma_ab, tau_a, tau_b, spin_scaling = False):


        uniform_factor = (3/10)*(3*np.pi**2)**(2/3)

        def l_1(rho):
            return rho**(1/3)

        def l_2(rho, gamma):
            return torch.sqrt(gamma)/(2*(3*np.pi**2)**(1/3)*rho**(4/3)+self.epsilon)

        def l_3(rho, gamma, tau):
            return (tau - gamma/(8*(rho+self.epsilon)))/(uniform_factor*rho**(5/3)+self.epsilon)

        if not spin_scaling:
            zeta = (rho0_a - rho0_b)/(rho0_a + rho0_b + self.epsilon)
            spinscale = 0.5*((1+zeta)**(4/3) + (1-zeta)**(4/3)) # zeta

        if self.level > 0:
            if spin_scaling:
                # descr1 = torch.log((2*rho0_a)**(1/3) + self.loge)
                descr1 = torch.log(l_1(2*rho0_a) + self.loge)
                # descr2 = torch.log((2*rho0_b)**(1/3) + self.loge)
                descr2 = torch.log(l_1(2*rho0_b) + self.loge)
            else:
                # descr1 = torch.log((rho0_a + rho0_b)**(1/3) + self.loge)# rho
                descr1 = torch.log(l_1(rho0_a + rho0_b) + self.loge)# rho
                descr2 = torch.log(spinscale) # zeta
            descr = torch.cat([descr1.unsqueeze(-1), descr2.unsqueeze(-1)],dim=-1)
        if self.level > 1:
            if spin_scaling:
                # descr3a = torch.sqrt(4*gamma_a)/(2*(3*np.pi**2)**(1/3)*(2*rho0_a)**(4/3)+self.epsilon) # s
                descr3a = l_2(2*rho0_a, 4*gamma_a) # s
                # descr3b = torch.sqrt(4*gamma_b)/(2*(3*np.pi**2)**(1/3)*(2*rho0_b)**(4/3) +self.epsilon) # s
                descr3b = l_2(2*rho0_b, 4*gamma_b) # s
                descr3 = torch.cat([descr3a.unsqueeze(-1), descr3b.unsqueeze(-1)],dim=-1)
                descr3 = (1-torch.exp(-descr3**2/self.s_gam))*torch.log(descr3 + 1)
            else:
                # descr3 = torch.sqrt(gamma_a + gamma_b + 2*gamma_ab)/(2*(3*np.pi**2)**(1/3)*(rho0_a + rho0_b)**(4/3)+self.epsilon) # s
                descr3 = l_2(rho0_a + rho0_b, gamma_a + gamma_b + 2*gamma_ab) # s
                descr3 = descr3/((1+zeta)**(2/3) + (1-zeta)**2/3)
                descr3 = descr3.unsqueeze(-1)
                descr3 = (1-torch.exp(-descr3**2/self.s_gam))*torch.log(descr3 + 1)
            descr = torch.cat([descr, descr3],dim=-1)
        if self.level == 3:
            if spin_scaling:
                # descr4a = (2*tau_a - 4*gamma_a/(16*(rho0_a+self.epsilon)))/(uniform_factor*(2*rho0_a)**(5/3)+self.epsilon)
                descr4a = l_3(2*rho0_a, 4*gamma_a, 2*tau_a)
                # descr4b = (2*tau_b - 4*gamma_b/(16*(rho0_b+self.epsilon)))/(uniform_factor*(2*rho0_b)**(5/3)+self.epsilon)
                descr4b = l_3(2*rho0_b, 4*gamma_b, 2*tau_b)
                descr4 = torch.cat([descr4a.unsqueeze(-1), descr4b.unsqueeze(-1)],dim=-1)
                descr4 = descr4**3/(descr4**2+self.epsilon)
            else:
                # descr4 = (tau_a + tau_b - (gamma_a + gamma_b+2*gamma_ab)/(8*(rho0_a + rho0_b + self.epsilon)))/(self.epsilon+(rho0_a + rho0_b)**(5/3)*((1+zeta)**(5/3) + (1-zeta)**(5/3))) # tau
                descr4 = l_3(rho0_a + rho0_b, gamma_a + gamma_b + 2*gamma_ab, tau_a + tau_b)
                descr4 = descr4/((1+zeta)**(5/3) + (1-zeta)**(5/3))
                descr4 = descr4**3/(descr4**2+self.epsilon)
                descr4 = descr4.unsqueeze(-1)
            descr4 = torch.log((descr4 + 1)/2)
            # descr4 = torch.log(descr4 + self.loge)
            # descr4 = (descr4)/(1+descr4)-0.5
            descr = torch.cat([descr, descr4],dim=-1)
        if spin_scaling:
            descr = descr.view(descr.size()[0],-1,2).permute(2,0,1)

        return descr

    def get_V(self, dm, scaling):
        v1, v2 = torch.autograd.functional.jacobian(self.forward, (dm,dm), create_graph=True)
        if dm.dim() == 3:
            v1 = v1 * scaling[0].unsqueeze(0)
            v2 = v2 * scaling[1].unsqueeze(0)
        else:
            v1 = v1 * scaling[0]
            v2 = v2 * scaling[1]
        return v1 + v2
#         return v1

    def forward(self, dm, dm1=None):
        Exc = 0
        if self.grid_models or self.heg_mult:
            if self.ao_eval.dim()==2:
                ao_eval = self.ao_eval.unsqueeze(0)
            else:
                ao_eval = self.ao_eval

            if self.training:
                noise = torch.abs(torch.randn(dm.size(),device=dm.device)*1e-8)
                noise = noise + torch.transpose(noise,-1,-2)
            else:
                noise = torch.zeros_like(dm)
#                 dm += noise
#             rho = .5*(torch.einsum('ij,xik,jk->xi', self.ao_eval[0], self.ao_eval, dm) +
#                   torch.einsum('xij,ik,jk->xi', self.ao_eval, self.ao_eval[0], dm))
            rho = torch.einsum('xij,yik,...jk->xy...i', ao_eval, ao_eval, dm + noise )
            if dm1 is None:
                dm1 = dm

            rho2 = torch.einsum('xij,yik,...jk->xy...i', ao_eval[1:], ao_eval[1:], dm1 + noise)
#             rho2 = rho[1:,1:,...]
#             if self.training:
#                 noise = torch.abs(torch.randn(rho[:,:,0].size(),device=rho.device)*1e-8)
#                 for i in range(rho.size()[-2]):
# #                     if not torch.all(rho[0,0,i]== torch.zeros_like(rho[0,0,i])):
#                     rho[:,:,i] += noise
#                     rho2[:,:,i] += noise[1:,1:]

            rho0 = rho[0,0]
            drho = rho[0,1:4] + rho[1:4,0]
#             tau = 0.5*(rho[1,1] + rho[2,2] + rho[3,3])
            tau = 0.5*(rho2[1,1] + rho2[2,2] + rho2[0,0])
            if dm.dim() == 3:
                rho0_a = rho0[0]
                rho0_b = rho0[1]

                gamma_a, gamma_b = torch.einsum('ij,ij->j',drho[:,0],drho[:,0]), torch.einsum('ij,ij->j',drho[:,1],drho[:,1])
                gamma_ab = torch.einsum('ij,ij->j',drho[:,0],drho[:,1])
                tau_a, tau_b = tau
            else:
                rho0_a = rho0_b = rho0*0.5
                gamma_a=gamma_b=gamma_ab= torch.einsum('ij,ij->j',drho[:],drho[:])*0.25
                tau_a = tau_b = tau*0.5

            exc = self.eval_grid_models(torch.cat([rho0_a.unsqueeze(-1),
                                                    rho0_b.unsqueeze(-1),
                                                    gamma_a.unsqueeze(-1),
                                                    gamma_ab.unsqueeze(-1),
                                                    gamma_b.unsqueeze(-1),
                                                    torch.zeros_like(gamma_a.unsqueeze(-1)),
                                                    torch.zeros_like(gamma_a.unsqueeze(-1)),
                                                    tau_a.unsqueeze(-1),
                                                    tau_b.unsqueeze(-1)],dim=-1))

            Exc += torch.sum(((rho0_a + rho0_b)*exc[:,0])*self.grid_weights)
        if self.nxc_models:
            for nxc_model in self.nxc_models:
                Exc += nxc_model(dm, self.ml_ovlp)


        return Exc

    def eval_grid_models(self, rho):
        Exc = 0
        rho0_a = rho[:, 0]
        rho0_b = rho[:, 1]
        gamma_a = rho[:, 2]
        gamma_ab = rho[:, 3]
        gamma_b = rho[:, 4]
        lapl_a = rho[:, 5]
        lapl_b = rho[:, 6]
        tau_a = rho[:, 7]
        tau_b = rho[:, 8]


        zeta = (rho0_a - rho0_b)/(rho0_a + rho0_b + 1e-8)
        rs = (4*np.pi/3*(rho0_a+rho0_b + 1e-8))**(-1/3)
        rs_a = (4*np.pi/3*(rho0_a + 1e-8))**(-1/3)
        rs_b = (4*np.pi/3*(rho0_b + 1e-8))**(-1/3)
        exc_a = torch.zeros_like(rho0_a)
        exc_b = torch.zeros_like(rho0_a)
        exc_ab = torch.zeros_like(rho0_a)

        spinscale = 0.5*((1+zeta)**(4/3) + (1-zeta)**(4/3)) # zeta
        if self.polynomial:
            descr_method = self.get_descriptors_pol
        else:
            descr_method = self.get_descriptors


        descr_dict = {}
        rho_tot = rho0_a + rho0_b
        if self.grid_models:

            for grid_model in self.grid_models:
                if not grid_model.spin_scaling:
                    if not 'c' in descr_dict:
                        descr_dict['c'] = descr_method(rho0_a, rho0_b, gamma_a, gamma_b,
                                                                         gamma_ab, tau_a, tau_b, spin_scaling = False)
                        descr_dict['c'] = descr_method(rho0_a, rho0_b, gamma_a, gamma_b,
                                                                         gamma_ab, tau_a, tau_b, spin_scaling = False)
                    descr = descr_dict['c']

                    exc = grid_model(descr,
                                      grid_coords = self.grid_coords,
                                      edge_index = self.edge_index)

                    if exc.dim() == 2: #If using spin decomposition
                        pw_alpha = self.pw_model(rs_a, torch.ones_like(rs_a))
                        pw_beta = self.pw_model(rs_b, torch.ones_like(rs_b))
                        pw = self.pw_model(rs, zeta)
                        ec_alpha = (1 + exc[:,0])*pw_alpha*rho0_a/rho_tot
                        ec_beta =  (1 + exc[:,1])*pw_beta*rho0_b/rho_tot
                        ec_mixed = (1 + exc[:,2])*(pw*rho_tot - pw_alpha*rho0_a - pw_beta*rho0_b)/rho_tot
                        exc_ab = ec_alpha + ec_beta + ec_mixed
                    else:
                        if self.pw_mult:
                            exc_ab += (1 + exc)*self.pw_model(rs, zeta)
                        else:
                            exc_ab += exc
                else:
                    if not 'x' in descr_dict:
                        descr_dict['x'] = descr_method(rho0_a, rho0_b, gamma_a, gamma_b,
                                                                         gamma_ab, tau_a, tau_b, spin_scaling = True)
                    descr = descr_dict['x']


                    exc = grid_model(descr,
                                  grid_coords = self.grid_coords,
                                  edge_index = self.edge_index)


                    if self.heg_mult:
                        exc_a += (1 + exc[0])*self.heg_model(2*rho0_a)*(1-self.exx_a)
                    else:
                        exc_a += exc[0]*(1-self.exx_a)

                    if torch.all(rho0_b == torch.zeros_like(rho0_b)): #Otherwise produces NaN's
                        exc_b += exc[0]*0
#                         print("hydrogen")
                    else:
                        if self.heg_mult:
                            exc_b += (1 + exc[1])*self.heg_model(2*rho0_b)*(1-self.exx_a)
                        else:
                            exc_b += exc[1]*(1-self.exx_a)

        else:
            if self.heg_mult:
                exc_a = self.heg_model(2*rho0_a)
                exc_b = self.heg_model(2*rho0_b)
            if self.pw_mult:
                exc_ab = self.pw_model(rs, zeta)

        exc = rho0_a/rho_tot*exc_a + rho0_b/rho_tot*exc_b + exc_ab

        return exc.unsqueeze(-1)

class NXC(torch.nn.Module):

    def __init__(self, ml_basis, symmetrizer, n_hidden=8, e_mean = 0, e_std = 1, device='cpu'):
        super().__init__()
        self.ml_net =  torch.nn.Sequential(
                            symmetrizer,
                            torch.nn.Linear(ml_basis, n_hidden),
                            torch.nn.GELU(),
                            torch.nn.Linear(n_hidden, n_hidden),
                            torch.nn.GELU(),
                            torch.nn.Linear(n_hidden, 1),
                            Reduce(e_mean, e_std, device)
                        ).double().to(device)

    def forward(self, dm, ml_ovlp):
        coeff = torch.einsum('ij,ijlk->lk', dm, ml_ovlp)
        return self.ml_net(coeff)

class C_L(torch.nn.Module):
    def __init__(self, n_input=2,n_hidden=16, device='cpu', use=[], ueg_limit=False):
        super().__init__()
        self.spin_scaling = False
        self.lob = False
        self.ueg_limit = ueg_limit

        if not use:
            self.use = torch.Tensor(np.arange(n_input)).long().to(device)
        else:
            self.use = torch.Tensor(use).long().to(device)
        self.net = torch.nn.Sequential(
                torch.nn.Linear(n_input, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, 1),
                torch.nn.Softplus()
            ).double().to(device)
        self.sig = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.lobf = LOB(2.0)
    def forward(self, rho, **kwargs):

        # squeezed = -self.net(rho[...,self.use]).squeeze()*self.gate(rho).squeeze()
        squeezed = -self.net(rho).squeeze()
        if self.ueg_limit:
            # ueg_lim = rho[...,self.use[0]]
            ueg_lim = self.tanh(rho[...,self.use[0]])
            if len(self.use) > 1:
                ueg_lim_a = torch.pow(self.tanh(rho[...,self.use[1]]),2)
            else:
                ueg_lim_a = 0

            return -self.lobf(-squeezed*(ueg_lim+ueg_lim_a))

        else:
            return -self.lobf(-squeezed)

class XC_L_POL(torch.nn.Module):

    def __init__(self, max_order=4, device='cpu', ueg_limit=True, lob=1.804, use=[]):
        super().__init__()
        self.spin_scaling = True
        self.lob = lob
        self.n_input = len(use)
        if ueg_limit:
            self.n_freepar = (max_order+1)**self.n_input - 1
            self.pars = torch.nn.Parameter(torch.Tensor([0.804] + [0.0]*(self.n_freepar-1)))
        else:
            self.n_freepar = (max_order+1)**self.n_input
            self.pars = torch.nn.Parameter(torch.Tensor([0.0, 0.804] + [0.0]*(self.n_freepar - 2)))
        self.pars.requires_grad = True
        self.max_order = max_order
        self.use = use
        self.gamma_s = torch.nn.Parameter(torch.Tensor([0.2730])) #PBE parameters
        self.gamma_s.requires_grad=True
        self.min_power = 1 if ueg_limit else 0

    def gen_features(self, inp):
        if self.n_input == 1:
            inp = (self.gamma_s*inp**2)/(1+self.gamma_s*inp**2)
            return torch.cat([inp**i for i in range(self.min_power, self.max_order+1)],dim=-1)
        elif self.input == 2:
            assert False
        else:
            assert False

    def forward(self, rho, **kwargs):
        inp = self.gen_features(rho[...,self.use])
        # pars = torch.abs(self.pars)
        if self.lob:
            pars = self.pars/torch.sum(self.pars)*(self.lob-1)
        else:
            pars = self.pars
        results = torch.einsum('i,...i', pars, inp)
        return results.squeeze( )

class C_L_POL(torch.nn.Module):

    def __init__(self, max_order=4, device='cpu',ueg_limit=True, use=[]):
        super().__init__()
        self.spin_scaling = False
        self.n_input = int(len(use)/2)
        if ueg_limit:
            self.n_freepar = (max_order)*(max_order+1)**(self.n_input - 1)
        else:
            self.n_freepar = (max_order+1)*(max_order+1)**(self.n_input - 1)
        # self.pars_ss = torch.nn.Linear(self.n_freepar, 1, bias=False)
        self.pars_ss = torch.nn.Parameter(torch.Tensor([0.1]*(self.n_freepar)))
        # self.pars_os = torch.nn.Linear(self.n_freepar, 1, bias=False)
        self.pars_os = torch.nn.Parameter(torch.Tensor([0.1]*(self.n_freepar)))
        self.max_order = max_order
        self.use = use
        self.gamma_s = torch.nn.Parameter(torch.Tensor([0.2730])) #PBE parameters
        self.gamma_rho = torch.nn.Parameter(torch.Tensor([0.02])) #PBE parameters
        # self.pars.weight.data.fill_(([0.804]+[0]*(self.n_freepar-1))) # PBE parameters
        self.gamma_s.requires_grad=True
        self.gamma_rho.requires_grad=True
        self.min_power = 1 if ueg_limit else 0
    def gen_features(self, inp):

        inp_ss_s = (torch.abs(self.gamma_s)*inp[:,2:]**2)/(1+torch.abs(self.gamma_s)*inp[:,2:]**2)
        inp_ss_rho = (torch.abs(self.gamma_rho)*inp[:,:2]**2)/(1+torch.abs(self.gamma_rho)*inp[:,:2]**2)
        os_s = (inp[:,3:4]**2 + inp[:,2:3]**2)*.5
        os_rho = (inp[:,0:1]**2 + inp[:,1:2]**2)*.5
        inp_os_s = (torch.abs(self.gamma_s)*os_s)/(1+torch.abs(self.gamma_s)*os_s)
        inp_os_rho = (torch.abs(self.gamma_rho)*os_rho)/(1+torch.abs(self.gamma_rho)*os_rho)
        return (torch.stack([inp_ss_s**i*inp_ss_rho**j for i in range(self.min_power, self.max_order+1) for j in range(0, self.max_order+1)], dim=-1),
                torch.cat([inp_os_s**i*inp_os_rho**j for i in range(self.min_power, self.max_order+1) for j in range(0, self.max_order+1)], dim=-1))

    def forward(self, rho, **kwargs):
        inp_ss, inp_os = self.gen_features(rho[...,self.use])
        # pars_ss = torch.abs(self.pars_ss)
        pars_ss = self.pars_ss
        pars_ss = pars_ss/torch.sum(pars_ss)
        # pars_os = torch.abs(self.pars_os)
        pars_os = self.pars_os
        pars_os = pars_os/torch.sum(pars_os)
        e_ss = torch.einsum('i,...i',pars_ss, inp_ss)
        e_os = torch.einsum('i,...i',pars_os, inp_os).unsqueeze(-1)
        res = -torch.cat([e_ss,e_os],dim=-1)
        return res


class XC_L(torch.nn.Module):
    def __init__(self, n_input=2,n_hidden=16, device='cpu', ueg_limit=False, lob=1.804, use=[]):
        super().__init__()
        self.ueg_limit = ueg_limit
        self.spin_scaling = True
        self.lob = lob

        if not use:
            self.use = torch.Tensor(np.arange(n_input)).long().to(device)
        else:
            self.use = torch.Tensor(use).long().to(device)
        self.net =  torch.nn.Sequential(
                torch.nn.Linear(n_input, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, 1),
            ).double().to(device)

        self.tanh = torch.nn.Tanh()
        self.sig = torch.nn.Sigmoid()
        self.lobf = LOB(lob)
        self.shift = 1/(1+np.exp(-1e-3))

    def forward(self, rho, **kwargs):
        squeezed = self.net(rho[...,self.use]).squeeze()
        if self.ueg_limit:
            ueg_lim = rho[...,self.use[0]]
            # ueg_lim = self.tanh(rho[...,self.use[0]])
            if len(self.use) > 1:
                ueg_lim_a = torch.pow(self.tanh(rho[...,self.use[1]]),2)
            else:
                ueg_lim_a = 0
        else:
            ueg_lim = 1
            ueg_lim_a = 0

        if self.lob:
            return self.lobf(squeezed*(ueg_lim+ueg_lim_a))
        else:
            return squeezed*(ueg_lim+ueg_lim_a)


class LOB(torch.nn.Module):

    def __init__(self, limit=1.804):
        super().__init__()
        self.sig = torch.nn.Sigmoid()
        self.limit = limit

    def forward(self, x):
         return self.limit*self.sig(x-np.log(self.limit-1))-1


class LDA_X(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, rho, **kwargs):
        return -3/4*(3/np.pi)**(1/3)*rho**(1/3)





# The following section was adapted from LibXC-4.3.4

params_a_pp     = [1,  1,  1]
params_a_alpha1 = [0.21370,  0.20548,  0.11125]
params_a_a      = [0.031091, 0.015545, 0.016887]
params_a_beta1  = [7.5957, 14.1189, 10.357]
params_a_beta2  = [3.5876, 6.1977, 3.6231]
params_a_beta3  = [1.6382, 3.3662,  0.88026]
params_a_beta4  = [0.49294, 0.62517, 0.49671]
params_a_fz20   = 1.709921


class PW_C(torch.nn.Module):


    def forward(self, rs, zeta):
        def g_aux(k, rs):
            return params_a_beta1[k]*torch.sqrt(rs) + params_a_beta2[k]*rs\
          + params_a_beta3[k]*rs**1.5 + params_a_beta4[k]*rs**(params_a_pp[k] + 1)

        def g(k, rs):
            return -2*params_a_a[k]*(1 + params_a_alpha1[k]*rs)\
          * torch.log(1 +  1/(2*params_a_a[k]*g_aux(k, rs)))

        def f_zeta(zeta):
            return ((1+zeta)**(4/3) + (1-zeta)**(4/3) - 2)/(2**(4/3)-2)

        def f_pw(rs, zeta):
          return g(0, rs) + zeta**4*f_zeta(zeta)*(g(1, rs) - g(0, rs) + g(2, rs)/params_a_fz20)\
          - f_zeta(zeta)*g(2, rs)/params_a_fz20

        return f_pw(rs, zeta)
