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

CONFIDENT = int(sys.argv[1])

for img in imgs:
     
    path, fileName = os.path.split(img)
    fileName_main = fileName.replace('.jpg','')

    # if fileName_main in open('results.txt').read():
    #     continue
    # TODO: need if-else for confident/frequent
    for r in range(10):
        random_class = np.random.randint(1599)+1
        file_output = fileName_main+'_'+str(random_class) +'.npy'

        if CONFIDENT:
            os.system('python ./code/pickobject_confident.py ' + img + ' ' + str(10000) + ' ' + 'True' +
                      ' ' + str(random_class) + ' ' + file_output + ' ' + str(gpuID))
        else:
            os.system('python ./code/pickobject_frequent.py '+ img +' '+ str(10000) + ' ' + 'True'+
                      ' ' + str(random_class) + ' ' + file_output + ' ' + str(gpuID))

        i = i+1

end = time.time()

print 'total time: ', end - start
