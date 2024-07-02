import cv2
import time
import numpy as np
import h5py
# File to read pictures from
pictures_file = "pictures.hdf5"
pic_key = "pics"
sit_key = "sitting"
# Font Settings
font_size = 0.6
font_pos = (10, 10) # Distance from top left in pixels
font_color = (0, 100, 0)  # BGR Color
font_weight = 2


f = h5py.File(pictures_file, "r")
start_pos = 0
if sit_key in f.keys():
	start_pos = len(f[sit_key])
dset = f[pic_key]
tot_pics = len(dset)

tags = []

exit_flag = False
i = 0
while i < tot_pics - start_pos:
    image = dset[start_pos + i]

    cv2.putText(image, "Hit S for sitting, D for just being a dog",
                font_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                font_color, font_weight)
    cv2.imshow("dog", image)
    
    key_input = cv2.waitKey(5)
    if key_input == ord("s"):
        tags.append(True)
        i += 1
    if key_input == ord("d"):
        tags.append(False)
        i += 1    
    if key_input == ord("q"):
        break
tags = np.array(tags, dtype=bool)
if sit_key in f.keys():
    f.create_dataset(sit_key, data=tags, chunks=True, maxshape=(None,))
else:
    dset.resize(dset.shape[0] + len(tags), axis=0)
    dset[start_pos:] = tags

cv2.destroyAllWindows()
del f