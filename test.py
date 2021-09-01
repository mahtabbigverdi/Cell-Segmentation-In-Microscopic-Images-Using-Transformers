import torch
import os
l = os.listdir('data/validation/labels')
s = 0
mydict = {}
ins = torch.load('CELL/temp/3_0.pt')
for ins_p in ins:
    #lab1 = len(torch.load('data/validation/labels/'+el).tolist())
    #lab2 = len(torch.load('data_cell/validation/instances/'+el).tolist())
    #lab3 = len(torch.load('data/validation/bbox/'+el).tolist())
    #print(ins_p[0])
    for m, n in ins_p[0]:
        print(m, "    ", n)
    break
#print(s/2)

    
