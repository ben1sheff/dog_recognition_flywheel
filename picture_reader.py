import cv2
import time
import numpy as np
import h5py
# File to read pictures from
pictures_file = "pictures.hdf5"
dset_name = "pics"
# Font Settings
font_size = 0.6
font_pos = (10, 20) # Distance from top left in pixels
font_color = (0, 100, 0)  # BGR Color
font_weight = 2


f = h5py.File(pictures_file, "r")
picture_ind = 0 # int(np.random.random() * len(f[dset_name]))
print("so far we have", len(f[dset_name]), "pictures")
image = f[dset_name][picture_ind]
cv2.putText(image, "dog " + str(picture_ind),
            font_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, 
            font_color, font_weight)
cv2.imshow("picture", image)
exit_flag = False
while not exit_flag:
    key_input = cv2.waitKey(5)
    if key_input == ord("q"):
        exit_flag = True
    if key_input == ord("n"):
        picture_ind = picture_ind + 1  #int(np.random.random() * len(f[dset_name]))
        image = f[dset_name][picture_ind]
        cv2.putText(image, "dog " + str(picture_ind),
                    font_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                    font_color, font_weight)
        cv2.imshow("picture", image)

cv2.destroyAllWindows()
del f