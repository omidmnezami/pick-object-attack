
import os
import numpy as np
import glob
import time
import sys

# you can change the gpuID as your device
gpuID = 0

# load the list of selected images
imgs = np.load('./selected_images.npy')

i = 0

start = time.time()

for img in imgs:
     random_class = 'untargeted'
     path, fileName = os.path.split(img)
     fileName_main = fileName.replace('.jpg','')

     # if fileName_main in open('results_untargeted.txt').read():
     #    continue

     file_output = fileName_main+'_'+str(random_class) +'.npy'
     os.system('python ./code/pickobject_confident_nontargeted.py '+ img +' '+ str(10000) + ' ' + 'False'+
               ' ' + random_class + ' ' + file_output + ' ' + str(gpuID))
     i = i+1

end = time.time()

print 'total time: ', end - start
