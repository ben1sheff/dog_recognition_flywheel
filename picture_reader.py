import cv2
import time
import numpy as np
import h5py
# File to read pictures from
pictures_file = "stream_data.hdf5"  # "pictures.hdf5"
dset_name = "pics"
sit_key = "sitting"
# Font Settings
font_size = 0.6
font_pos = (10, 20) # Distance from top left in pixels
font_color = (0, 100, 0)  # BGR Color
font_weight = 2


f = h5py.File(pictures_file, "r+")
dset = f[dset_name]
tags = f[sit_key]
print("so far we have", len(f[dset_name]), "pictures")

num_tags = 0
if sit_key in f.keys():
    num_tags = len(f[sit_key])

key_input = ord("n")
picture_ind = -1 # int(np.random.random() * len(f[dset_name]))
exit_flag = False
re_save_flag = False
while not exit_flag:
    if key_input == ord("c"):
        dset = np.concat([dset[:picture_ind], dset[picture_ind+1:]])  # check for end-of-file errors
        tags = np.concat([tags[:picture_ind], tags[picture_ind+1:]])
        key_input = ord("n")
        picture_ind -= 1
    if key_input == ord("s"):
        re_save_flag = True
        key_input = ord("q")
    if key_input == ord("q"):
        exit_flag = True
    if key_input == ord("b"):
        picture_ind -= 2
        key_input = ord("n")
    if key_input == ord("n"):
        picture_ind = (picture_ind + 1) % len(dset)  #int(np.random.random() * len(f[dset_name]))
        image = dset[picture_ind]

        # Add the tags if we have them
        if picture_ind < num_tags:
            tag_text = "dog is standing"
            if tags[picture_ind]:
                tag_text = "dog is sitting"
            cv2.putText(image, tag_text,
                        (font_pos[0], font_pos[1] + 30),
                         cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                        font_color, font_weight)
        cv2.putText(image, "dog " + str(picture_ind) + " n: next, b: back, c:cut, s: save, q: quit",
                    font_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                    font_color, font_weight)
        cv2.imshow("picture", image)
    key_input = cv2.waitKey(5)

cv2.destroyAllWindows()
if re_save_flag:
    f[dset_name].resize(dset.shape[0], axis=0)
    f[dset_name][:] = dset[:]
    sit_set = f[sit_key]
    f[sit_key].resize(tags.shape[0], axis=0)
    f[sit_key][:] = tags[:]
    print("overwriting data")
del f