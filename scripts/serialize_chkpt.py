import torch
from dpyscf.net import *
import os
import sys
import importlib
import json



DEVICE='cpu'
torch.set_default_dtype(torch.double)
config = json.loads(open(sys.argv[2],'r').read())
if sys.argv[1] == '0':
    path = None
else:
    path = sys.argv[1]
xc = get_scf(config['type'], config['pretrain_loc'], config['hyb_par'], path = path,
             polynomial=config.get('polynomial',False),
             ueg_limit = not config.get('free',False)).xc

if len(sys.argv) > 4:
    print('Only using grid model {:d}'.format(int(sys.argv[4])))
    model = int(sys.argv[4])
    xc.grid_models = xc.grid_models[model:model+1]
xc.evaluate()
xc.forward = xc.eval_grid_models
traced = torch.jit.trace(xc, torch.abs(torch.rand(100,9)))
os.mkdir(sys.argv[3])

torch.jit.save(traced, sys.argv[3] +'/xc')
print('EXX', xc.exx_a.detach().numpy())
with open(sys.argv[3] +'/exx_a', 'w') as file:
    file.write('{:.5f}'.format(xc.exx_a.detach().numpy()[0]))
