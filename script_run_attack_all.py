import os
import numpy as np
import glob
import time
import sys


# you can change the gpuID as your device
gpuID = 0

# load the path list of selected images
# (We selected these images from the dev set of MSCOCO. You need to change the path list according to your local paths.)
imgs = np.load('./selected_images.npy')

i = 0

start = time.time()

for img in imgs:
     random_class = 'untargeted_all'
     path, fileName = os.path.split(img)
     fileName_main = fileName.replace('.jpg','')

     file_output = fileName_main+'_'+str(random_class) +'.npy'
     os.system('python ./code/attack_all.py '+ img +' '+ str(10000) + ' ' + 'False'+ ' ' + random_class +
               ' ' + file_output + ' ' + str(gpuID))
     i = i+1

end = time.time()

print 'total time: ', end - start
