
import os
import numpy as np
import glob
import time
import sys


gpuID = int(sys.argv[1])

if gpuID == 1:
    imgs = np.load('./selected_images.npy')[:300]
else:
    imgs = np.load('./selected_images.npy')[300+350*(gpuID-2):300+350*(gpuID-1)]


i = 0

start = time.time()

for img in imgs:
     random_class = 'untargeted_all'
     path, fileName = os.path.split(img)
     fileName_main = fileName.replace('.jpg','')

     file_output = fileName_main+'_'+str(random_class) +'.npy'
     os.system('python ./code/attackall_nontargeted.py '+ img +' '+ str(10000) + ' ' + 'False'+ ' ' + random_class +
               ' ' + file_output + ' ' + str(gpuID))
     i = i+1

end = time.time()

print 'total time: ', end - start
