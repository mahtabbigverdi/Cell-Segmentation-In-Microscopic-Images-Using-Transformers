import torch
import os
import numpy as np
dirs = os.listdir('CELL/temp')
res = None
for el in dirs:
    res = None
    #labels = torch.load('data/train/labels/' + el).tolist()
    ins = torch.load('CELL/temp/' + el)
    ins  = (ins > 0.5).float()
    ins = ins.detach().cpu().numpy()
    for s in range(ins.shape[0]) :
        #if labels[s] == 2:
        if res is None :
             res = ins[s]
        else:
             res += ins[s]

    print(el)
    print(np.unique(res))
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')


