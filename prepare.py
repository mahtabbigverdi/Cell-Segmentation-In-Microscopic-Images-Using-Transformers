import os
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import torch

l = os.listdir('train/y')
mydict = {}
for el in l:
    temp = el.split("_")
    if temp[0] in mydict:
        mydict[temp[0]].append(el)
    else:
        mydict[temp[0]] = [el]

for i, instances in mydict.items():
    ins = []
    for el in instances:
        f = os.path.join('train/y', el)
        img = Image.open(f)
        img = np.array(img)
        nuclei = np.where(img == 40, 1, 0)
        if nuclei.ndim == 3 :
            nuclei = nuclei[:,:,0]
        nuclei = nuclei.tolist()
        ins.append(nuclei)
        cell = np.where(img != 0, 1, 0)
        if cell.ndim == 3 :
            cell = cell[:,:,0]
        cell = cell.tolist()
        ins.append(cell)

    file_name = i +'.pt'
    print(file_name)
    ins = np.array(ins)
    ins = torch.Tensor(ins)
    torch.save(ins,'train/instances/'+file_name)


print('instances finished')

for i, instances in mydict.items():
    boxes = []
    labels = []
    for el in instances:
        f = os.path.join('train/y', el)
        img = Image.open(f)
        img = np.array(img)
        nuclei = np.where(img == 40, img, 0)
        if nuclei.ndim == 3 :
            nuclei = nuclei[:,:,0]
        pos_n = np.where(nuclei)
        xmin = np.min(pos_n[1])
        xmax = np.max(pos_n[1])
        ymin = np.min(pos_n[0])
        ymax = np.max(pos_n[0])

        labels.append(2)
        boxes.append([xmin, ymin, xmax, ymax])


        cell = np.where(img == 40, 20, img)
        if cell.ndim == 3 :
            cell = cell[:,:,0]
        pos_c = np.where(cell)
        xmin = np.min(pos_c[1])
        xmax = np.max(pos_c[1])
        ymin = np.min(pos_c[0])
        ymax = np.max(pos_c[0])

        labels.append(1)
        boxes.append([xmin, ymin, xmax, ymax])

    file_name = i +'.pt'
    labels = torch.Tensor(labels)
    torch.save(labels, 'train/labels/'+file_name)

    boxes = torch.Tensor(boxes)
    torch.save(boxes, 'train/bbox/'+file_name)





