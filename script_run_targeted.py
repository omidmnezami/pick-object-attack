
import os
import numpy as np
import glob
import time
import sys

gpuID = int(sys.argv[1])

if gpuID == 2:
    imgs = np.load('./selected_images.npy')[:1000]
else:
    imgs = np.load('./selected_images.npy')[300+350*(gpuID-2):300+350*(gpuID-1)]

i = 0

start = time.time()

for img in imgs:
     
    path, fileName = os.path.split(img)
    fileName_main = fileName.replace('.jpg','')

    # if fileName_main in open('results.txt').read():
    #     continue

    for r in range(10):
        random_class = np.random.randint(1599)+1
        file_output = fileName_main+'_'+str(random_class) +'.npy'
        os.system('python ./code/pickbox_frequent.py '+ img +' '+ str(10000) + ' ' + 'True'+
                  ' ' + str(random_class) + ' ' + file_output + ' ' + str(gpuID))
        i = i+1

end = time.time()

print 'total time: ', end - start
