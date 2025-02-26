from glob import glob
import os
import itertools
import numpy as np
if __name__ == '__main__':

    # To make a file (cam0, cam2, disp0, disp2)
    dataset_dir  = "./Train"
    maxd0 = []
    maxd2 = []
    l = glob(os.path.abspath(dataset_dir + "/*"))
    for i in l:

        #get cam0 cam2
        cam0 = glob(os.path.abspath(i + "/camera_00/*.png"))
        cam2 = glob(os.path.abspath(i + "/camera_02/*.png"))
        d0 = np.load(os.path.abspath(i + "/disp_00.npy")).ravel()
        d2 = np.load(os.path.abspath(i + "/disp_02.npy")).ravel()

        maxd0.append(np.max(d0))

        maxd2.append(np.max(d2))

        print(np.mean(maxd0), np.mean(maxd2))