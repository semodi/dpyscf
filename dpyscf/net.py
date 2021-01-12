import torch
from torch.nn import Sequential as Seq, Linear, ReLU,Sigmoid,GELU
from torch_geometric.nn import MessagePassing
import pyscf
from pyscf import gto,dft,scf
import torch
import numpy as np
import scipy
from ase import Atoms
from ase.io import read
from .torch_routines import *



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
    
    def __init__(self, grid_models=None, nxc_models=None, heg_mult=True, level = 1, model_mult=[], exx_a=None):
        super().__init__()
        self.grid_models = None
        self.nxc_models = None
        self.heg_mult = heg_mult
        self.edge_index = None
        self.grid_coords = None
        self.training = True
        self.level = level
        self.epsilon = 1e-7
        self.loge = 1e-3
        self.s_gam = 1

        if heg_mult:
            self.heg_model = LDA_X()
        
        if grid_models:
            if not isinstance(grid_models, list): grid_models = [grid_models]
            self.grid_models = torch.nn.ModuleList(grid_models)
        if nxc_models:
            if not isinstance(nxc_models, list): grid_models = [nxc_models]
            self.nxc_models = torch.nn.ModuleList(nxc_models)
        if model_mult:
            assert len(model_mult) == len(grid_models)
            self.register_buffer('model_mult',torch.Tensor(model_mult))
        else:
            self.model_mult = [1 for m in self.grid_models]
        if exx_a is not None:
            self.register_buffer('exx_a', torch.Tensor([exx_a]))
        
    def evaluate(self):
        self.training=False
    def train(self):
        self.training=True
     
    def add_model_mult(self, model_mult):
        del(self.model_mult)
        self.register_buffer('model_mult',torch.Tensor(model_mult))
    
    def add_exx_a(self, exx_a):
        self.register_buffer('exx_a', torch.Tensor([exx_a]))
      
    
#     def get_descriptors(self, rho0_a, rho0_b, gamma_a, gamma_b, gamma_ab, tau_a, tau_b, spin_scaling = False):
      
#         uniform_factor = (3/10)*(3*np.pi**2)**(2/3)
#         if not spin_scaling:
#             zeta = (rho0_a - rho0_b)/(rho0_a + rho0_b + epsilon)
     
#             spinscale = 0.5*((1+zeta)**(4/3) + (1-zeta)**(4/3)) # zeta
           
#         if self.level > 0:
#             if spin_scaling:
#                 descr1 = (2*rho0_a)**(1/3)
#                 descr2 = (2*rho0_b)**(1/3)
#             else: 
#                 descr1 = (rho0_a + rho0_b)**(1/3) # rho
#                 descr2 = spinscale # zeta
#             descr = torch.cat([descr1.unsqueeze(-1), descr2.unsqueeze(-1)],dim=-1)
#         if self.level > 1:
#             if spin_scaling:
#                 descr3a = torch.sqrt(4*gamma_a)/(2*(3*np.pi**2)**(1/3)*descr1**4+epsilon) # s
#                 descr3b = torch.sqrt(4*gamma_b)/(2*(3*np.pi**2)**(1/3)*descr2**4+epsilon) # s
#                 descr = torch.cat([descr, descr3a.unsqueeze(-1), descr3b.unsqueeze(-1)],dim=-1)
#             else:
#                 descr3 = torch.sqrt(gamma_a + gamma_b + 2*gamma_ab)/(2*(3*np.pi**2)**(1/3)*descr1**4+epsilon) # s
#                 descr = torch.cat([descr, descr3.unsqueeze(-1)],dim=-1)
#         if self.level == 3:
#             if spin_scaling:
# #                 descr4a = (2*tau_a)/(descr1**5+epsilon) # tau
# #                 descr4b = (2*tau_b)/(descr2**5+epsilon) # tau
# #                 descr4a = (2*tau_a - 4*gamma_a/(16*rho0_a+epsilon))/(uniform_factor*2*descr1**5*1/2**(5/3)+epsilon)
# #                 descr4b = (2*tau_b - 4*gamma_b/(16*rho0_b+epsilon))/(uniform_factor*2*descr2**5*1/2**(5/3)+epsilon)
      
# #                 descr4a = ((2*tau_a - 4*gamma_a/(16*(rho0_a+1e-8)))/(uniform_factor*2*(descr1)**5*1/2**(5/3)+1e-8) + 1)/2
# #                 descr4b = ((2*tau_b - 4*gamma_b/(16*(rho0_b+1e-8)))/(uniform_factor*2*(descr2)**5*1/2**(5/3)+1e-8) + 1)/2
#                 descr4a = ((2*tau_a - 4*gamma_a/(16*(rho0_a+epsilon)))/(uniform_factor*2*(descr1)**5*1/2**(5/3)+epsilon) + 1)/2
#                 descr4b = ((2*tau_b - 4*gamma_b/(16*(rho0_b+epsilon)))/(uniform_factor*2*(descr2)**5*1/2**(5/3)+epsilon) + 1)/2
# #         
#                 descr = torch.cat([descr, descr4a.unsqueeze(-1), descr4b.unsqueeze(-1)],dim=-1)
#             else:
# #                 descr4 = (tau_a + tau_b)/(epsilon+descr1**5*((1+zeta)**(5/3) + (1-zeta)**(5/3))) # tau
#                 descr4 = ((tau_a + tau_b - (gamma_a + gamma_b+2*gamma_ab)/(8*(rho0_a + rho0_b+epsilon)))/(epsilon+descr1**5*((1+zeta)**(5/3) + (1-zeta)**(5/3))) +1)/2 # tau
            
#                 descr = torch.cat([descr, descr4.unsqueeze(-1)],dim=-1)
#         if spin_scaling:
# #             descr = descr.view(descr.size()[0],2,-1).transpose(0,1)
#             descr = descr.view(descr.size()[0],-1,2).permute(2,0,1)
# #             print(descr.size())
#         return torch.log(descr+1e-3)

    def get_descriptors(self, rho0_a, rho0_b, gamma_a, gamma_b, gamma_ab, tau_a, tau_b, spin_scaling = False):
      
        uniform_factor = (3/10)*(3*np.pi**2)**(2/3)
        if not spin_scaling:
            zeta = (rho0_a - rho0_b)/(rho0_a + rho0_b + self.epsilon)
     
            spinscale = 0.5*((1+zeta)**(4/3) + (1-zeta)**(4/3)) # zeta
           
        if self.level > 0:
            if spin_scaling:
                descr1 = torch.log((2*rho0_a)**(1/3) + self.loge)
                descr2 = torch.log((2*rho0_b)**(1/3) + self.loge)
            else: 
                descr1 = torch.log((rho0_a + rho0_b)**(1/3) + self.loge)# rho
                descr2 = spinscale # zeta
            descr = torch.cat([descr1.unsqueeze(-1), descr2.unsqueeze(-1)],dim=-1)
        if self.level > 1:
            if spin_scaling:
                descr3a = torch.sqrt(4*gamma_a)/(2*(3*np.pi**2)**(1/3)*(2*rho0_a)**(4/3)+self.epsilon) # s
                descr3b = torch.sqrt(4*gamma_b)/(2*(3*np.pi**2)**(1/3)*(2*rho0_b)**(4/3) +self.epsilon) # s
                descr3 = torch.cat([descr3a.unsqueeze(-1), descr3b.unsqueeze(-1)],dim=-1)
            else:
                descr3 = torch.sqrt(gamma_a + gamma_b + 2*gamma_ab)/(2*(3*np.pi**2)**(1/3)*(rho0_a + rho0_b)**(4/3)+self.epsilon) # s
                descr3 = descr3.unsqueeze(-1)
            descr3 = (1-torch.exp(-descr3**2/self.s_gam))*torch.log(descr3 + 1)
#             descr3 = torch.log(descr3 + self.loge)
            descr = torch.cat([descr, descr3],dim=-1)
            
        if self.level == 3:
            if spin_scaling:
#                 descr4a = (2*tau_a - 4*gamma_a/(16*(rho0_a+self.epsilon)))/(uniform_factor*2*(2*rho0_a)**(5/3)*1/2**(5/3)+1e-6)
#                 descr4b = (2*tau_b - 4*gamma_b/(16*(rho0_b+self.epsilon)))/(uniform_factor*2*(2*rho0_b)**(5/3)*1/2**(5/3)+1e-6) 
                
                descr4a = (2*tau_a - 4*gamma_a/(16*(rho0_a+self.epsilon)))/(uniform_factor*(2*rho0_a)**(5/3)+self.epsilon)
                descr4b = (2*tau_b - 4*gamma_b/(16*(rho0_b+self.epsilon)))/(uniform_factor*(2*rho0_b)**(5/3)+self.epsilon)
                
                descr4 = torch.cat([descr4a.unsqueeze(-1), descr4b.unsqueeze(-1)],dim=-1)
#                 descr4 = descr4**3/(descr4**2+self.epsilon)
                descr4 = descr4**3/(descr4**2+self.epsilon)
            else:
                descr4 = ((tau_a + tau_b - (gamma_a + gamma_b+2*gamma_ab)/(8*(rho0_a + rho0_b + self.epsilon)))/(self.epsilon+(rho0_a + rho0_b)**(5/3)*((1+zeta)**(5/3) + (1-zeta)**(5/3))) +1)/2 # tau
#                 descr4 = ((tau_a + tau_b)/(self.epsilon+(rho0_a + rho0_b)**(5/3)*((1+zeta)**(5/3) + (1-zeta)**(5/3))) +0)/1 # tau
                
                descr4 = descr4**3/(descr4**2+self.epsilon)
                descr4 = descr4.unsqueeze(-1)
#             descr4[descr4<0.0] = descr4[descr4<0.0]*0
            descr4 = torch.log((descr4 + 1)/2)
#             descr4 = torch.log(descr4+self.loge)
#             descr4 = torch.log(descr4+self.loge)
            descr = torch.cat([descr, descr4],dim=-1)
        if spin_scaling:
#             descr = descr.view(descr.size()[0],2,-1).transpose(0,1)
            descr = descr.view(descr.size()[0],-1,2).permute(2,0,1)
#             print(descr.size())

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
#             rho = .5*(torch.einsum('ij,xik,jk->xi', self.ao_eval[0], self.ao_eval, dm) +
#                   torch.einsum('xij,ik,jk->xi', self.ao_eval, self.ao_eval[0], dm))
            rho = torch.einsum('xij,yik,...jk->xy...i', ao_eval, ao_eval, dm ) 
            if dm1 is None:
                dm1 = dm
            
            rho2 = torch.einsum('xij,yik,...jk->xy...i', ao_eval[1:], ao_eval[1:], dm1) 
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
                
            if self.training:
                noise = torch.abs(torch.randn(rho0_a.size(),device=rho.device)*1e-8)
                rho0_a += noise
                rho0_b += noise
                  
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
        exc_a = torch.zeros_like(rho0_a)
        exc_b = torch.zeros_like(rho0_a)
        exc_ab = torch.zeros_like(rho0_a)

        spinscale = 0.5*((1+zeta)**(4/3) + (1-zeta)**(4/3)) # zeta

        descr_dict = {}

        if self.grid_models:

            for grid_model, m in zip(self.grid_models, self.model_mult):
                if not grid_model.spin_scaling:
                    if not 'c' in descr_dict:
                        descr_dict['c'] = self.get_descriptors(rho0_a, rho0_b, gamma_a, gamma_b,
                                                                         gamma_ab, tau_a, tau_b, spin_scaling = False)
                    descr = descr_dict['c']

                    exc_ab += m*grid_model(descr, 
                                      grid_coords = self.grid_coords,
                                      edge_index = self.edge_index)

                else:
                    if not 'x' in descr_dict:
                        descr_dict['x'] = self.get_descriptors(rho0_a, rho0_b, gamma_a, gamma_b,
                                                                         gamma_ab, tau_a, tau_b, spin_scaling = True)
                    descr = descr_dict['x']


                    exc = grid_model(descr, 
                                  grid_coords = self.grid_coords,
                                  edge_index = self.edge_index)

                    
                    if self.heg_mult:
                        exc_a += (1 + exc[0])*self.heg_model(2*rho0_a)*m
                    else:
                        exc_a += exc[0]*m
                        
                    if torch.all(rho0_b == torch.zeros_like(rho0_b)): #Otherwise produces NaN's
                        exc_b += exc[0]*0
#                         print("hydrogen")
                    else:
                        if self.heg_mult:
                            exc_b += (1 + exc[1])*self.heg_model(2*rho0_b)*m
                        else:
                            exc_b += exc[1]*m


#         if self.heg_mult:
# #                 exc_ = (1+exc)*self.heg_model(rho0_a + rho0_b)*spinscale
#             exc_a = (1+exc_a)*self.heg_model(2*rho0_a)  
#             exc_b = (1+exc_b)*self.heg_model(2*rho0_b)
#             if 'c' in descr_dict:    
#                 exc_ab = (1+exc_ab)*self.heg_model(rho0_a+rho0_b)

        rho_tot = rho0_a + rho0_b
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
    def __init__(self, n_input=2,n_hidden=16, device='cpu', use=[]):
        super().__init__()
        self.spin_scaling = False
        self.lob = False

        if not use:
            self.use = torch.Tensor(np.arange(n_input)).long().to(device)
        else:
            self.use = torch.Tensor(use).long().to(device)
        self.net =  torch.nn.Sequential(
                torch.nn.Linear(len(use), n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden, 1),
                torch.nn.Softplus()
            ).double().to(device)
        self.gate = torch.nn.Sequential(
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
    
        
    def forward(self, rho, **kwargs):
        return -self.net(rho[...,self.use]).squeeze()*self.gate(rho).squeeze()
#         return -self.gate(rho).squeeze()
class XC_L(torch.nn.Module):
    def __init__(self, n_input=2,n_hidden=16, device='cpu', spin_scaling=False, lob=False, use=[]):
        super().__init__()
        self.spin_scaling = spin_scaling
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
        self.lobf = LOB()
        self.shift = 1/(1+np.exp(-1e-3))
        
    def forward(self, rho, **kwargs):
        
        
        if self.spin_scaling and self.lob:
            
            squeezed = self.net(rho[...,self.use]).squeeze()
#             ueg_lim = self.sig(rho[...,self.use[0]]) - self.shift
            ueg_lim = rho[...,self.use[0]]
            if len(self.use) > 1:
                ueg_lim_a = torch.pow(self.tanh(rho[...,self.use[1]]),2)
            else:
                ueg_lim_a = 0

            return self.lobf(squeezed*(ueg_lim+ueg_lim_a))
#             elif len(self.use) == 1:
#                 return (self.tanh((self.net(rho[...,self.use])).squeeze()) * 1.0865 + 0.0865)*(self.sig(torch.sum(rho[...,self.use] ,dim=-1)))

        else:
            return (self.net(rho[...,self.use])).squeeze()
        
class LOB(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.sig = torch.nn.Sigmoid()
    def forward(self, x):
         return 2.17*self.sig(x-np.log(1.17))-1   
    
class LDA_X(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, rho, **kwargs):
        return -3/4*(3/np.pi)**(1/3)*rho**(1/3)


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, n_gaussians = 10, r_cut=0.1, n_hidden=12, device='cpu'):
        super(EdgeConv, self).__init__(aggr='add') 
        self.mlp = Seq(Linear(10 + n_hidden, 12),
                       GELU(),
                       Linear(n_hidden, n_hidden),
                       GELU(),
                       Linear(n_hidden, 1)
                      ).to(device)
        
        self.centers = torch.linspace(0, r_cut, n_gaussians).unsqueeze(0).to(device)
        self.sigma = r_cut/n_gaussians*2
        self.centers_rho = torch.linspace(-10, 2, 5).unsqueeze(0).to(device)
        self.sigma_rho = 2
        self.r_cut = r_cut
        
    def forward(self, rho, edge_index, pos):
        propagated =  self.propagate(edge_index, x=rho, pos=pos) 
        return propagated
    
    def eval_gaussians(self, dr):
        return torch.exp(-((dr-self.centers)/self.sigma)**2/2)
    
    def eval_gaussians_rho(self, rho):
        return torch.exp(-((rho-self.centers_rho)/self.sigma_rho)**2/2)
    
    def message(self, x_i, x_j,pos_i,pos_j):

        dr =  torch.norm(pos_j - pos_i,dim=-1,keepdim=True)
        dr_embedded = self.eval_gaussians(dr)
        
        tmp = torch.cat([self.eval_gaussians_rho(x_i[:,:1]+x_j[:,:1]),
                         self.eval_gaussians_rho(-torch.abs(x_i[:,:1]-x_j[:,:1])), 
                         dr_embedded], dim=1)  # tmp has shape [E, 2 * in_channels]
        
        fc = 1-(.5*(1-torch.cos(np.pi*(dr)/self.r_cut)))**8
        result = self.mlp(tmp)*fc*torch.exp(x_j[:,:1])**3
    
        return result*x_j[:,1:]
