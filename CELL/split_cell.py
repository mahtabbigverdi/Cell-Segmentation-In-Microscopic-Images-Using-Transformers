import os
import numpy as np

l = os.listdir('data/validation/x')

for index in range(len(l)):
    file_name = 'data_cell/train/x/' + l[index]
    dest = 'data_cell/validation/x/'
    os.system('mv '+file_name+ ' '+dest)
    
    
    file_name = 'data_cell/train/bbox/' + l[index].split('.')[0] + '.pt'
    dest = 'data_cell/validation/bbox/'
    os.system('mv '+file_name+ ' '+dest)

    file_name = 'data_cell/train/labels/' + l[index].split('.')[0] + '.pt'
    dest = 'data_cell/validation/labels/'
    os.system('mv '+file_name+ ' '+dest)

    file_name = 'data_cell/train/instances/' + l[index].split('.')[0] + '.pt'
    dest = 'data_cell/validation/instances/'
    os.system('mv '+file_name+ ' '+dest)
