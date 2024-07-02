import cv2
import time
import numpy as np
import h5py
# File to read pictures from
pictures_file = "pictures.hdf5"
dset_name = "pics"
sit_key = "sitting"
# Font Settings
font_size = 0.6
font_pos = (10, 20) # Distance from top left in pixels
font_color = (0, 100, 0)  # BGR Color
font_weight = 2


f = h5py.File(pictures_file, "r")
print("so far we have", len(f[dset_name]), "pictures")

num_tags = 0
if sit_key in f.keys():
    num_tags = len(f[sit_key])

key_input = ord("n")
picture_ind = -1 # int(np.random.random() * len(f[dset_name]))
exit_flag = False
while not exit_flag:
    if key_input == ord("q"):
        exit_flag = True
    if key_input == ord("n"):
        picture_ind = (picture_ind + 1) % len(f[dset_name])  #int(np.random.random() * len(f[dset_name]))
        image = f[dset_name][picture_ind]

        # Add the tags if we have them
        if picture_ind < num_tags:
            tag_text = "dog is standing"
            if f[sit_key][picture_ind]:
                tag_text = "dog is sitting"
            cv2.putText(image, tag_text,
                        (font_pos[0], font_pos[1] + 30),
                         cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                        font_color, font_weight)
        cv2.putText(image, "dog " + str(picture_ind),
                    font_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                    font_color, font_weight)
        cv2.imshow("picture", image)
    key_input = cv2.waitKey(5)

cv2.destroyAllWindows()
del f