import os
import numpy as np

l = os.listdir('data/train/x')
indexes = np.random.choice(np.arange(0, len(l)), len(l) // 10 , replace=False)
for index in indexes:
    file_name = 'data/train/x/' + l[index]
    dest = 'data/validation/x/'
    os.system('mv '+file_name+ ' '+dest)
    
    
    file_name = 'data/train/bbox/' + l[index].split('.')[0] + '.pt'
    dest = 'data/validation/bbox/'
    os.system('mv '+file_name+ ' '+dest)

    file_name = 'data/train/labels/' + l[index].split('.')[0] + '.pt'
    dest = 'data/validation/labels/'
    os.system('mv '+file_name+ ' '+dest)

    file_name = 'data/train/instances/' + l[index].split('.')[0] + '.pt'
    dest = 'data/validation/instances/'
    os.system('mv '+file_name+ ' '+dest)
