import torch
from dpyscf.net import *
import os
import sys
import importlib




DEVICE='cpu'
torch.set_default_dtype(torch.double)



xc = torch.load(sys.argv[1]).xc
xc.evaluate()
xc.forward = xc.eval_grid_models
traced = torch.jit.trace(xc, torch.abs(torch.rand(100,9)))
os.mkdir(sys.argv[2])

torch.jit.save(traced, sys.argv[2] +'/xc')
# print('EXX', xc.exx_a.detach().numpy())
# with open(sys.argv[2] +'/exx_a', 'w') as file:
#     file.write('{:.5f}'.format(xc.exx_a.detach().numpy()[0]))
