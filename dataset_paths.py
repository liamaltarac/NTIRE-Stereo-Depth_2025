from glob import glob
import os

if __name__ == '__main__':

    dataset_dir  = "./dataset/test_mono_nogt"

    l = glob(os.path.abspath(dataset_dir + "/*"))

    with open(os.path.abspath("./dataset_paths/test.txt"), 'w') as f:
        for i in l:
            camera0 = glob(os.path.join(i, "camera_00/*.png"))
            for j in camera0:
                f.write(f"{j}\n")