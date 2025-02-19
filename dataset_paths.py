from glob import glob
import os
import itertools
if __name__ == '__main__':

    # To make a file (cam0, cam2, disp0, disp2)
    dataset_dir  = "./Train"

    l = glob(os.path.abspath(dataset_dir + "/*"))
    
    with open(os.path.abspath("./dataset_paths/train_stereo.txt"), 'w') as f:
        for i in l:

            #get cam0 cam2
            cam0 = glob(os.path.abspath(i + "/camera_00/*.png"))
            cam2 = glob(os.path.abspath(i + "/camera_02/*.png"))
            d0 = glob(os.path.abspath(i + "/disp_00.npy"))[0]
            d2 = glob(os.path.abspath(i + "/disp_02.npy"))[0]

            for c0, c2 in itertools.product(cam0, cam2):   
                f.write(f"{c0}, {c2}, {d0}, {d2}\n")


    #print(l)

    '''with open(os.path.abspath("./dataset_paths/test.txt"), 'w') as f:
        for i in l:
            camera0 = glob(os.path.join(i, "camera_00/*.png"))
            for j in camera0:
                f.write(f"{j}\n")'''