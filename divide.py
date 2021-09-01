import numpy as np
import os
from PIL import Image
import torch

for el in os.listdir('cytoplasm/data/validation/x'):
    name = el.split('.')[0]
    main_img = Image.open('data/validation/x/'+ el)
    main_img = np.array(main_img)
    imglist = []
    step = 300
    w = 600
    h = 600
    for i in range(0,main_img.shape[0], step):
        if main_img.shape[0] - (i+step) < 300:
            end = True
        else:
            end = False
        for j in range(0,main_img.shape[1], step):
            if  main_img.shape[1] - (j+step) < 300:
                if end:
                    imglist.append([i,main_img.shape[0],j,main_img.shape[1]])
                else:
                    imglist.append([i,i+h,j,main_img.shape[1]])
                break
            else:
                if end:
                    imglist.append([i,main_img.shape[0],j,j+w])
                else:
                    imglist.append([i,i+h,j,j+w])

        if end:
            break
    ins = torch.load('cytoplasm/data/validation/instances/'+ name+'.pt')
    l = torch.load('cytoplasm/data/validation/labels/'+ name+'.pt').tolist()
    b = torch.load('cytoplasm/data/validation/bbox/'+ name+'.pt').tolist()
    #print(ins.shape)
    for i,s in enumerate(imglist):
        boxes = []
        labels =[]
        new_ins = []
        t = torch.Tensor(s)
        torch.save(t,'data/validation/new_x/'+name+"_"+str(i+1)+".pt")
        ins_i = ins[:,s[0]:s[1], s[2]:s[3]]
        number_ins = ins_i.shape[0]
        for index in range(number_ins):
            temp = ins_i[index]
            if 1 in temp:
                new_ins.append(temp.tolist())
                labels.append(l[index])
                pos = np.where(temp)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
                

        if len(new_ins)==0:
            os.remove('data/validation/new_x/'+name+"_"+str(i+1)+".pt")
            continue
        new_ins = np.array(new_ins)
        new_ins = torch.Tensor(new_ins)
        torch.save(new_ins,'data/validation/new_instances/'+name+"_"+str(i+1)+".pt")
        torch.save(torch.Tensor(labels),'data/validation/new_labels/'+name+"_"+str(i+1)+".pt")
        torch.save(torch.Tensor(boxes),'data/validation/new_bbox/'+name+"_"+str(i+1)+".pt")


    print(el,main_img.shape,len(imglist))
