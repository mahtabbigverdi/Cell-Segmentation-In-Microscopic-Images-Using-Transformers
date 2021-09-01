import torch
from PIL import Image as im
import numpy as np
ins = torch.load('data/train/instances/9674.pt')[1].numpy()[500: , 1300:]
torch.save(torch.Tensor(ins),'temp.pt')
#data = im[].fromarray(ins).convert('RGB')
#print(np.unique(ins))
#print(ins)
#data.save('temp.png')


