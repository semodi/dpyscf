import torch
import numpy as np
def ip_loss(results, loss, **kwargs):
    e_ip = results['e_ip'].unsqueeze(-1)
    e_ip_ref = results['e_ip_ref']
    lip = loss(e_ip, e_ip_ref)
    return lip

def energy_loss(results, loss, **kwargs):
    E = results['E']
    E_ref = results['E_ref']
    weights = kwargs.get('weights', torch.linspace(0,1,E.size()[0])**2).to(results['E'].device)
    skip_steps = kwargs.get('skip_steps',0)

    dE = weights*(E_ref - E)
    dE = dE[skip_steps:]
    ldE = loss(dE, torch.zeros_like(dE))
    return ldE

def econv_loss(results, loss, **kwargs):
    E = results['E']
    weights = kwargs.get('weights', torch.linspace(0,1,E.size()[0])**2).to(results['E'].device)
    skip_steps = kwargs.get('skip_steps',0)

    dE = weights*(E - E[-1])
    dE = dE[skip_steps:]
    ldE = loss(dE, torch.zeros_like(dE))
    return ldE

def ae_loss(ref_dict,pred_dict, loss, **kwargs):
    ref = torch.cat(list(atomization_energies(ref_dict).values()))
    pred = torch.cat(list(atomization_energies(pred_dict).values()))
    assert len(ref) == 1
    ref = ref.expand(pred.size()[0])
    if pred.size()[0] > 1:
        weights = kwargs.get('weights', torch.linspace(0,1,pred.size()[0])**2).to(pred.device)
    else:
        weights = 1
    lae = loss((ref-pred)*weights,torch.zeros_like(pred))
    return lae


def dm_loss(results, loss, **kwargs):
    fcenter = results['fcenter'][0]
    dm = results['dm']
    dm_ref = results['dm_ref'][0]
    ddm = torch.einsum('ijkl,ij,kl',fcenter,dm,dm) +\
                torch.einsum('ijkl,ij,kl',fcenter,dm_ref,dm_ref) -\
                2*torch.einsum('ijkl,ij,kl',fcenter,dm,dm_ref)
    ldm = loss(ddm/results['n_elec'][0,0]**2, torch.zeros_like(ddm))
    return ldm

def rho_loss(results, loss, **kwargs):
    rho_ref = results['rho'][0]
    ao_eval = results['ao_eval'][0]
    dm = results['dm']
    if dm.ndim == 2:
        rho = torch.einsum('ij,ik,jk->i',
                           ao_eval[0], ao_eval[0], dm)
        drho = torch.sqrt(torch.sum((rho-rho_ref)**2*results['grid_weights'])/results['n_elec'][0,0]**2)
#         drho = (rho-rho_ref)*torch.sqrt(results['grid_weights'])/results['n_elec'][0,0]
        lrho = loss(drho, torch.zeros_like(drho))

    else:
        rho = torch.einsum('ij,ik,xjk->xi',
                           ao_eval[0], ao_eval[0], dm)
        if torch.sum(results['mo_occ']) == 1:
            drho = torch.sqrt(torch.sum((rho[0]-rho_ref[0])**2*results['grid_weights'])/torch.sum(results['mo_occ'][0,0])**2)
        else:
            drho = torch.sqrt(torch.sum((rho[0]-rho_ref[0])**2*results['grid_weights'])/torch.sum(results['mo_occ'][0,0])**2 +\
                   torch.sum((rho[1]-rho_ref[1])**2*results['grid_weights'])/torch.sum(results['mo_occ'][0,1])**2)

#         drho = torch.cat([(rho[0]-rho_ref[0])*torch.sqrt(results['grid_weights'])/results['n_elec'][0,0],
#                           (rho[1]-rho_ref[1])*torch.sqrt(results['grid_weights'])/results['n_elec'][0,0]])

        lrho = loss(drho, torch.zeros_like(drho))
    return lrho

def moe_loss(results, loss, **kwargs):
    dmoe = results['mo_energy_ref'][0] - results['mo_energy']

    norbs = kwargs.get('norbs',-1)
    if norbs > -1:
        dmoe = dmoe[:norbs]

    lmoe = loss(dmoe, torch.zeros_like(dmoe))
    return lmoe

def gap_loss(results, loss, nhomo, **kwargs):
    ref = results['mo_energy_ref'][:,nhomo+1] - results['mo_energy_ref'][:,nhomo]
    pred = results['mo_energy'][nhomo+1] - results['mo_energy'][nhomo]
    dgap = ref - pred
    lgap = loss(dgap, torch.zeros_like(dgap))
    return lgap

def atomization_energies(energies):
    def split(el):
        import re
        res_list = [s for s in re.split("([A-Z][^A-Z]*)", el) if s]
        return res_list


    ae = {}
    for key in energies:
        if isinstance(energies[key],torch.Tensor):
            if len(split(key)) == 1:continue
            e_tot = energies[key].clone()
        else:
            e_tot = np.array(energies[key])
        for symbol in split(key):
            e_tot -= energies[symbol]
        ae[key] = e_tot
    return ae
