import torch
from net import *
import os
import sys
import importlib
from code_snapshot import get_scf



DEVICE='cpu'
torch.set_default_dtype(torch.double)

# xc = get_scf(sys.argv[1]).xc

xc = torch.load(path)
xc.grid_models = xc.grid_models[0:1]
xc.evaluate()
xc.forward = xc.eval_grid_models
traced = torch.jit.trace(xc, torch.abs(torch.rand(100,9)))
os.mkdir(sys.argv[2])

torch.jit.save(traced, sys.argv[2] +'/xc')
# print('EXX', xc.exx_a.detach().numpy())
# with open(sys.argv[2] +'/exx_a', 'w') as file:
#     file.write('{:.5f}'.format(xc.exx_a.detach().numpy()[0]))
