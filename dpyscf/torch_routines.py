import torch
import scipy
from opt_einsum import contract



def get_hcore(v, t):
    return v + t

class get_veff(torch.nn.Module):
    def __init__(self, xc=False, model=None, df= False):
        super().__init__()
        self.xc = xc
        self.model = model
        if df:
            self.forward = self.forward_df

    def forward(self, dm, eri):
        J = contract('...ij,ijkl->...kl',dm, eri)
        if not self.xc:
            K = self.model.exx_a * contract('...ij,ikjl->...kl',dm, eri)
        else:
            K =  torch.zeros_like(J)

        if J.ndim == 3:
            return J[0] + J[1] - K
        else:
            return J-0.5*K
    
    def forward_df(self, dm, df_3c, df_2c_inv, eri):
        
        J = contract('mnQ, QP, ...ij, ijP->...mn', df_3c, df_2c_inv, dm, df_3c)
        if not self.xc:
            K = self.model.exx_a * contract('...ij,ikjl->...kl',dm, eri)
        else:
            K =  torch.zeros_like(J)

        if J.ndim == 3:
            return J[0] + J[1] - K
        else:
            return J-0.5*K
        
    
def get_fock(hc, veff):
    return hc + veff


class eig_scipy(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, h, s):
        '''Solver for generalized eigenvalue problem

        .. math:: HC = SCE
        '''
        h = h.detach().numpy()
        s = s.detach().numpy()
        e, c = scipy.linalg.eigh(h,s)
        c = torch.from_numpy(c)
        e = torch.from_numpy(e)
#         c = torch.mm(s_inv_oh, c)
        return e, c

class eig(torch.nn.Module):

    def __init__(self):
        super().__init__()
    def forward(self, h, s_oh, s_inv_oh):
        '''Solver for generalized eigenvalue problem

        .. math:: HC = SCE
        '''
#         e, c = torch.symeig(contract('ij,jk,kl->il',s_inv_oh, h, s_inv_oh), eigenvectors=True,upper=False)
#         c = torch.mm(s_inv_oh, c)
        e, c = torch.symeig(contract('ij,...jk,kl->...il',s_oh, h, s_inv_oh), eigenvectors=True,upper=False)
        c = contract('ij,...jk ->...ik',s_inv_oh, c)
        return e, c



class energy_tot(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, dm, hcore, veff):
        return torch.sum((contract('...ij,ij', dm, hcore) + .5*contract('...ij,...ij', dm, veff))).unsqueeze(0)

class make_rdm1(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, mo_coeff, mo_occ):
        if mo_coeff.ndim == 3:
            mocc_a = mo_coeff[0, :, mo_occ[0]>0]
            mocc_b = mo_coeff[1, :, mo_occ[1]>0]
            if torch.sum(mo_occ[1]) > 0:
                return torch.stack([contract('ij,jk->ik', mocc_a*mo_occ[0,mo_occ[0]>0], mocc_a.T),
                                    contract('ij,jk->ik', mocc_b*mo_occ[1,mo_occ[1]>0], mocc_b.T)],dim=0)
            else:
                return torch.stack([contract('ij,jk->ik', mocc_a*mo_occ[0,mo_occ[0]>0], mocc_a.T),
                                    torch.zeros_like(mo_coeff)[0]],dim=0)
        else:
            mocc = mo_coeff[:, mo_occ>0]
            return contract('ij,jk->ik', mocc*mo_occ[mo_occ>0], mocc.T)

def diis(deltadm):

    deltadm = torch.stack(deltadm)
    delta = deltadm.view(deltadm.size()[0],-1)
    delta = torch.mm(delta,delta.T)
    delta = torch.cat([delta,torch.ones_like(delta)[:,:1]],dim=-1)
    delta = torch.cat([delta,torch.ones_like(delta)[:1,:]],dim=0)
    delta[-1,-1] = 0
    B = torch.ones_like(delta)[:,:1]
    B[:-1] = 0
    sol = torch.solve(B,delta)[0][:-1]

    return torch.sum(deltadm*sol.unsqueeze(-1),dim=0)

class SCF(torch.nn.Module):

    def __init__(self, alpha=0.8, nsteps=10, xc=None, device='cpu', exx=False, ncore=0):
        super().__init__()

        self.nsteps = nsteps
        self.alpha = alpha
        self.get_veff = get_veff(not exx, xc).to(device) # Include Fock (exact) exchange?

        self.eig = eig().to(device)
        self.energy_tot = energy_tot().to(device)
        self.make_rdm1 = make_rdm1().to(device)
        self.xc = xc
        self.ncore = int(ncore/2)

    def forward(self, dm, matrices, sc=True):
        dm = dm[0]

        # Required matrices
        v, t, mo_occ, s_oh, s_inv_oh,  e_nuc, s = [matrices[key][0] for key in \
                                             ['v','t','mo_occ','s_oh','s_inv_oh',
                                             'e_nuc','s']]

        # Optional matrices
        eri = matrices.get('eri',[None])[0]
        ml_ovlp = matrices.get('ml_ovlp',[None])[0]
        grid_weights = matrices.get('grid_weights',[None])[0]
        grid_coords = matrices.get('grid_coords',[None])[0]
        edge_index = matrices.get('edge_index',[None])[0]
        ao_eval = matrices.get('ao_eval',[None])[0]
        L = matrices.get('L', [torch.eye(dm.size()[-1])])[0]
        scaling = matrices.get('scaling',[torch.ones([dm.size()[-1]]*2)])[0]
        ip_idx = matrices.get('ip_idx', [0])[0]
        df_2c_inv = matrices.get('df_2c_inv',[None])[0] 
        df_3c = matrices.get('df_3c',[None])[0] 
        vh_on_grid = matrices.get('vh_on_grid',[None])[0]
        
        dm_old = dm

        E = []
        deltadm = []
        for step in range(self.nsteps):
            if len(deltadm) > 90 and step < self.nsteps - 5:
                dm = dm_old + diis(deltadm)
#             alpha = self.alpha
            else:
                alpha = (self.alpha)**(step)+0.3
#                 alpha = (self.alpha)**(step)+0.1
                beta = (1-alpha)
                dm = alpha * dm + beta * dm_old

#             if not step==0:
#                 deltadm.append(dm-dm_old)
#             deltadm = deltadm[-9:]

            dm_old = dm

            hc = get_hcore(v,t)
            if df_3c is not None:
                veff = self.get_veff.forward_df(dm, df_3c, df_2c_inv, eri)
            else:
                veff = self.get_veff(dm, eri)

            if self.xc:
                self.xc.ao_eval = ao_eval
                self.xc.grid_weights = grid_weights
                self.xc.grid_coords = grid_coords
                self.xc.edge_index = edge_index
                self.xc.ml_ovlp = ml_ovlp
                if vh_on_grid is not None:
                    self.xc.vh_on_grid = vh_on_grid
                    self.xc.df_2c_inv = df_2c_inv
                    self.xc.df_3c = df_3c
                    
                exc = self.xc(dm)

                
                vxc = torch.autograd.functional.jacobian(self.xc, dm, create_graph=True)
                if vxc.dim() > 2:
                    vxc = contract('ij,xjk,kl->xil',L,vxc,L.T)
                    vxc = torch.where(scaling.unsqueeze(0) > 0 , vxc, scaling.unsqueeze(0))
#                     vxc = vxc*scaling.unsqueeze(0)

                else:
                    vxc = torch.mm(L,torch.mm(vxc,L.T))
#                     vxc = vxc*scaling
                    vxc = torch.where(scaling > 0 , vxc, scaling)
    
                if torch.sum(mo_occ) == 1:   # Otherwise H produces NaNs
                    vxc[1] = torch.zeros_like(vxc[1])
                
                veff += vxc
                if self.xc.training:
                    
                    noise = torch.abs(torch.randn(vxc.size(),device=vxc.device)*1e-8)
#                         noise = torch.abs(torch.randn(vxc.size(),device=vxc.device)*1e-5)
                    noise = noise + torch.transpose(noise,-1,-2)
                    veff = veff + noise
#                     vxc = vxc*(1 + noise)
                
            else:
                exc=0
                vxc=torch.zeros_like(veff)

            
            f = get_fock(hc, veff)
            if sc:
                try:
                    mo_e, mo_coeff = self.eig(f, s_oh, s_inv_oh)

                except RuntimeError:
                    raise RuntimeError

                dm = self.make_rdm1(mo_coeff, mo_occ)
#             if self.xc:
#                 exc = self.xc(dm)
            E.append(self.energy_tot(dm_old, hc, veff-vxc)+e_nuc + exc)
            if not sc:
                break
#             E.append(self.energy_tot(dm, hc, veff-vxc)+e_nuc + exc)
        mo_occ[:self.ncore] = 0
#         dm = self.make_rdm1(mo_coeff, mo_occ)
#         if dm.dim() == 3:
#             e_ip = mo_e[0][ip_idx]
#         else:
#             e_ip = mo_e[ip_idx]
        e_ip = 0
        mo_e = 0
        results = {'E':torch.cat(E), 'dm':dm, 'mo_energy':mo_e,'e_ip':e_ip}

        return results
